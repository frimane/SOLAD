#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Created on Wed Jan 21 23:44:03 2026
# 
# @author: Azeddine Frimane
# 
# Implementation notes:
#   - FIX: min_count=1 replaced by count()-based NaN guard (pandas resample does not
#          support min_count on .mean())
#   - Full verbose logging at every stage
#   - Defensive assertions throughout
#   - NaN diagnostics per profile (pre-accept AND pre-reject)
#   - No theory changes: all logic, thresholds, and flow are identical to the original
#

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pvlib
from datetime import datetime, timedelta
import logging
import random

from data.feature_engineering import SolarFeatureEngineer


class SolarPreprocessor:

    def __init__(self, config: Dict, logger: logging.Logger):

        self.config = config
        self.logger = logger
        self.data_config = config['data']
        self.qc_config = config['data']['quality_control']

        # Station metadata: latitude, longitude, elevation (meters)
        # Source: SURFRAD official station documentation
        self.station_metadata = {
            'bon': {'lat': 40.05192,  'lon': -88.37309,   'elev': 230},
            'dra': {'lat': 36.62373,  'lon': -116.01947,  'elev': 1007},
            'fpk': {'lat': 48.30783,  'lon': -105.1017,   'elev': 634},
            'gwn': {'lat': 34.2547,   'lon': -89.8729,    'elev': 98},
            'psu': {'lat': 40.72012,  'lon': -77.93085,   'elev': 376},
            'sxf': {'lat': 43.73403,  'lon': -96.62328,   'elev': 473},
            'tbl': {'lat': 40.12498,  'lon': -105.2368,   'elev': 1689},
        }

        # ---------- extract configuration values ----------
        self.nighttime_zenith_threshold  = self.data_config['nighttime']['zenith_threshold_degrees']
        self.daytime_zenith_threshold    = self.qc_config['daytime_zenith_threshold']
        self.min_ghi_for_df              = self.data_config['min_ghi_for_df_calculation']
        self.min_clear_sky_for_csi       = self.data_config['min_clear_sky_for_csi_calculation']
        self.dif_ghi_tolerance           = self.qc_config['dif_ghi_tolerance']
        self.profile_half_window_hours   = self.data_config['profile_half_window_hours']
        self.solar_noon_tolerance_minutes = self.data_config['solar_noon_alignment_tolerance_minutes']
        self.air_mass_config             = self.data_config['air_mass']

        # Guard: critical thresholds must be positive
        assert self.nighttime_zenith_threshold > 0, (
            f"nighttime_zenith_threshold must be > 0, got {self.nighttime_zenith_threshold}"
        )
        assert self.daytime_zenith_threshold > 0, (
            f"daytime_zenith_threshold must be > 0, got {self.daytime_zenith_threshold}"
        )
        assert self.profile_half_window_hours > 0, (
            f"profile_half_window_hours must be > 0, got {self.profile_half_window_hours}"
        )
        assert self.min_ghi_for_df >= 0, (
            f"min_ghi_for_df must be >= 0, got {self.min_ghi_for_df}"
        )
        assert self.min_clear_sky_for_csi >= 0, (
            f"min_clear_sky_for_csi must be >= 0, got {self.min_clear_sky_for_csi}"
        )

        # Validate configuration before proceeding
        if config['data'].get('validation', {}).get('check_config_consistency', True):
            self._validate_config()

        self.logger.info("SolarPreprocessor initialized")
        self.logger.info(f"  Nighttime threshold          : {self.nighttime_zenith_threshold} degrees")
        self.logger.info(f"  Daytime QC threshold         : {self.daytime_zenith_threshold} degrees")
        self.logger.info(f"  Profile window               : +/- {self.profile_half_window_hours} hours around solar noon")
        self.logger.info(f"  Min GHI for diffuse fraction : {self.min_ghi_for_df} W/m^2")
        self.logger.info(f"  Min clear-sky for CSI        : {self.min_clear_sky_for_csi} W/m^2")
        self.logger.info(f"  DIF > GHI tolerance          : {self.dif_ghi_tolerance} W/m^2")
        self.logger.info(f"  Solar noon alignment tol.    : {self.solar_noon_tolerance_minutes} min")

        # Initialize feature engineer
        self.feature_engineer = SolarFeatureEngineer(config)
        self.derived_feature_names = self.feature_engineer.get_feature_list()

        self.logger.info(f"  Derived features             : {len(self.derived_feature_names)}")
        if self.derived_feature_names:
            self.logger.info(f"    {', '.join(self.derived_feature_names)}")

    # =========================================================================
    # CONFIGURATION VALIDATION
    # =========================================================================

    def _validate_config(self) -> None:
        #
        # Validate configuration for internal consistency.
        # 
        # Checks:
        #   - Profile length matches window configuration
        #   - Nowcasting timesteps match hours and resolution
        #   - Geometry features are properly categorized for normalization
        #   - Required threshold values are present
        # 
        # Raises:
        #     ValueError: If configuration is inconsistent
        #
        self.logger.info("Validating configuration consistency...")

        # -- Profile length ----------------------------------------------------
        expected_profile_length = int(
            (2 * self.profile_half_window_hours * 60) /
            self.data_config['target_resolution_minutes']
        )
        actual_profile_length = self.data_config['profile_length']

        if expected_profile_length != actual_profile_length:
            raise ValueError(
                f"Config mismatch: profile_length={actual_profile_length} "
                f"but 2 * profile_half_window_hours * 60 / target_resolution_minutes "
                f"= {expected_profile_length}"
            )
        self.logger.info(f"  profile_length check passed  : {actual_profile_length} timesteps")

        # -- Nowcasting window -------------------------------------------------
        nowcast_config         = self.data_config['nowcasting']
        expected_hist_timesteps = int(
            (nowcast_config['history_length_hours'] * 60) /
            self.data_config['target_resolution_minutes']
        )
        expected_fore_timesteps = int(
            (nowcast_config['forecast_length_hours'] * 60) /
            self.data_config['target_resolution_minutes']
        )

        if expected_hist_timesteps != nowcast_config['history_length_timesteps']:
            raise ValueError(
                f"Config mismatch: history_length_timesteps="
                f"{nowcast_config['history_length_timesteps']} "
                f"but history_length_hours * 60 / target_resolution_minutes "
                f"= {expected_hist_timesteps}"
            )
        if expected_fore_timesteps != nowcast_config['forecast_length_timesteps']:
            raise ValueError(
                f"Config mismatch: forecast_length_timesteps="
                f"{nowcast_config['forecast_length_timesteps']} "
                f"but forecast_length_hours * 60 / target_resolution_minutes "
                f"= {expected_fore_timesteps}"
            )

        expected_total = expected_hist_timesteps + expected_fore_timesteps
        if expected_total != nowcast_config['total_sequence_length']:
            raise ValueError(
                f"Config mismatch: total_sequence_length="
                f"{nowcast_config['total_sequence_length']} "
                f"but history + forecast = {expected_total}"
            )
        self.logger.info(
            f"  nowcasting window check passed: "
            f"hist={expected_hist_timesteps}, fore={expected_fore_timesteps}, "
            f"total={expected_total} timesteps"
        )

        # -- Geometry normalisation categorisation ----------------------------
        geometry_features = set(self.data_config['deterministic_geometry_features'])
        norm_config        = self.config['normalization']['geometry']
        cyclic_features    = set(norm_config['cyclic_features'])
        minmax_features    = set(norm_config['minmax_features'])
        all_norm_features  = cyclic_features | minmax_features

        if geometry_features != all_norm_features:
            missing = geometry_features - all_norm_features
            extra   = all_norm_features - geometry_features
            error_msg = "Geometry features mismatch between data config and normalization config:\n"
            if missing:
                error_msg += f"  Missing in normalization config : {missing}\n"
            if extra:
                error_msg += f"  Extra in normalization config   : {extra}\n"
            raise ValueError(error_msg)

        overlap = cyclic_features & minmax_features
        if overlap:
            raise ValueError(f"Features cannot be both cyclic and minmax: {overlap}")

        self.logger.info(
            f"  geometry normalisation check passed: "
            f"{len(cyclic_features)} cyclic, {len(minmax_features)} minmax"
        )
        self.logger.info("Configuration validation passed OK")

    # =========================================================================
    # PUBLIC ENTRY POINT
    # =========================================================================

    def process_station(self, station: str, year: int) -> List[Dict]:
        #
        # Process one year of data for one station.
        # 
        # Args:
        #     station: Station code (e.g., 'dra', 'bon')
        #     year   : Year to process
        # 
        # Returns:
        #     List of daily profile dictionaries, each containing:
        #       - date, station, solar_noon, timestamps
        #       - ghi, dif, csi, diffuse_fraction
        #       - air_mass, dni (observation channels)
        #       - deterministic_geometry, derived_features
        #
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Processing {station.upper()} - {year}")
        self.logger.info(f"{'='*60}")

        assert station in self.station_metadata, (
            f"Unknown station '{station}'. "
            f"Valid stations: {list(self.station_metadata.keys())}"
        )
        assert isinstance(year, int) and 1990 <= year <= 2100, (
            f"year must be an integer in [1990, 2100], got {year!r}"
        )

        # Stage 1: Load raw data
        self.logger.info(f"[Stage 1] Loading raw data ...")
        df = self._load_surfrad_data(station, year)
        if df is None or len(df) == 0:
            self.logger.warning(f"  No data for {station} {year} - skipping")
            return []
        self.logger.info(f"  Loaded {len(df):,} raw records")

        # Stage 2: Compute solar geometry
        self.logger.info(f"[Stage 2] Computing solar geometry ...")
        df = self._compute_solar_geometry(df, station)
        self.logger.info(f"  Solar geometry computed for {len(df):,} records")

        # Stage 3: Compute clear-sky irradiance
        self.logger.info(f"[Stage 3] Computing clear-sky irradiance (Ineichen) ...")
        df = self._compute_clear_sky(df, station)
        self.logger.info(
            f"  Clear-sky range: "
            f"[{df['ghi_clear_sky'].min():.1f}, {df['ghi_clear_sky'].max():.1f}] W/m^2"
        )

        # Stage 4: Handle nighttime (CRITICAL - must come before any division)
        self.logger.info(f"[Stage 4] Handling nighttime ...")
        df = self._handle_nighttime(df)

        # Stage 5: Apply quality control
        self.logger.info(f"[Stage 5] Applying quality control ...")
        n_before_qc = len(df)
        df = self._apply_quality_control(df)
        self.logger.info(
            f"  QC: {n_before_qc - len(df):,} records removed "
            f"({100*(n_before_qc-len(df))/max(n_before_qc,1):.2f}%), "
            f"{len(df):,} remain"
        )

        # Stage 6: Compute derived variables (CSI, diffuse fraction)
        self.logger.info(f"[Stage 6] Computing derived variables (CSI, diffuse fraction) ...")
        df = self._compute_derived_variables(df)

        # Stage 7: Resample to target resolution
        target_res = self.data_config['target_resolution_minutes']
        self.logger.info(f"[Stage 7] Resampling to {target_res}-minute resolution ...")
        n_before_resample = len(df)
        df = self._resample_data(df)
        self.logger.info(
            f"  Resampled: {n_before_resample:,} -> {len(df):,} records "
            f"at {target_res}-min resolution"
        )

        # Stage 8: Create solar-noon-centered profiles
        self.logger.info(f"[Stage 8] Creating solar-noon-centered profiles ...")
        profiles = self._create_solar_noon_centered_profiles(df, station)

        self.logger.info(f"  Created {len(profiles)} daily profiles for {station.upper()} {year}")
        self.logger.info(f"{'='*60}")

        return profiles

    # =========================================================================
    # STAGE 1: LOAD
    # =========================================================================

    def _load_surfrad_data(self, station: str, year: int) -> Optional[pd.DataFrame]:
        #
        # Load SURFRAD data for one station and year.
        # 
        # Args:
        #     station: Station code
        #     year   : Year to load
        # 
        # Returns:
        #     DataFrame with columns: timestamp (index), ghi, dif, dni, zenith
        #     Returns None if no data found
        #
        input_dir = Path(self.data_config['input_dir']) / station
        pattern   = f"{year}_{station}.csv"
        files     = sorted(input_dir.glob(pattern))

        if not files:
            self.logger.warning(
                f"  No files matching '{pattern}' in {input_dir}"
            )
            return None

        self.logger.info(f"  Found {len(files)} file(s) for {station} {year}")

        dfs = []
        for file in files:
            try:
                self.logger.debug(f"    Reading {file.name} ...")
                df_file = pd.read_csv(file)

                # Guard: required columns must exist
                required_cols = {'datetime', 'ghi', 'dhi', 'dni', 'solar_zenith'}
                missing_cols  = required_cols - set(df_file.columns)
                assert not missing_cols, (
                    f"File {file.name} is missing columns: {missing_cols}"
                )

                # Ensure timestamps are in UTC
                df_file['timestamp'] = pd.to_datetime(df_file['datetime'], utc=True)

                # Select and rename columns to standard names
                df_file = df_file[['timestamp', 'ghi', 'dhi', 'dni', 'solar_zenith']]
                df_file.columns = ['timestamp', 'ghi', 'dif', 'dni', 'zenith']

                # Guard: no completely empty rows in critical columns
                n_all_nan = df_file[['ghi', 'dif', 'dni']].isna().all(axis=1).sum()
                if n_all_nan > 0:
                    self.logger.warning(
                        f"    {file.name}: {n_all_nan} rows have all-NaN irradiance - "
                        f"these will be handled downstream"
                    )

                dfs.append(df_file)
                self.logger.debug(f"    {file.name}: {len(df_file):,} records loaded")

            except Exception as e:
                self.logger.warning(f"  Error loading {file}: {e}")
                continue

        if not dfs:
            self.logger.warning(f"  All files failed to load for {station} {year}")
            return None

        # Concatenate all files, sort, and set timestamp as index
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.set_index('timestamp')

        # Guard: no duplicate timestamps
        n_dupes = df.index.duplicated().sum()
        if n_dupes > 0:
            self.logger.warning(
                f"  {n_dupes} duplicate timestamps found - keeping first occurrence"
            )
            df = df[~df.index.duplicated(keep='first')]

        # Guard: index must be monotonic after dedup
        assert df.index.is_monotonic_increasing, (
            "Timestamp index is not monotonic after deduplication - check raw data"
        )

        self.logger.info(
            f"  Final loaded dataset: {len(df):,} records "
            f"from {df.index.min()} to {df.index.max()}"
        )

        return df

    # =========================================================================
    # STAGE 2: SOLAR GEOMETRY
    # =========================================================================

    def _compute_solar_geometry(self, df: pd.DataFrame, station: str) -> pd.DataFrame:
        #
        # Compute solar position and geometry features using pvlib.
        # 
        # Args:
        #     df     : DataFrame with timestamp index
        #     station: Station code
        # 
        # Returns:
        #     DataFrame with added geometry columns.
        #
        metadata = self.station_metadata[station]
        lat, lon, elev = metadata['lat'], metadata['lon'], metadata['elev']

        self.logger.debug(
            f"  Computing geometry for lat={lat}, lon={lon}, elev={elev}m"
        )

        location = pvlib.location.Location(
            latitude=lat,
            longitude=lon,
            altitude=elev,
            tz='UTC'
        )

        # -- Solar position ----------------------------------------------------
        solar_position = location.get_solarposition(df.index)
        df['solar_zenith_angle']  = solar_position['zenith']
        df['solar_azimuth_angle'] = solar_position['azimuth']
        df['solar_elevation']     = solar_position['elevation']

        # Guard: zenith should be in [0, 180]
        zenith_valid = df['solar_zenith_angle'].between(0, 180, inclusive='both')
        n_bad_zenith = (~zenith_valid).sum()
        if n_bad_zenith > 0:
            self.logger.warning(
                f"  {n_bad_zenith} records with solar_zenith_angle outside [0, 180] - "
                f"min={df['solar_zenith_angle'].min():.2f}, "
                f"max={df['solar_zenith_angle'].max():.2f}"
            )

        # -- Extraterrestrial radiation ----------------------------------------
        df['extraterrestrial_radiation'] = pvlib.irradiance.get_extra_radiation(df.index)

        # Guard: ETR should be positive (seasonal variation ~1320-1420 W/m^2)
        n_bad_etr = (df['extraterrestrial_radiation'] <= 0).sum()
        if n_bad_etr > 0:
            self.logger.warning(
                f"  {n_bad_etr} records with non-positive extraterrestrial_radiation"
            )

        # -- Air mass ---------------------------------------------------------
        airmass           = location.get_airmass(df.index)
        df['air_mass']    = airmass['airmass_relative']
        df['air_mass']    = df['air_mass'].replace([np.inf, -np.inf], np.nan)

        n_nan_am = df['air_mass'].isna().sum()
        self.logger.debug(
            f"  air_mass: {n_nan_am} NaN before nighttime handling "
            f"(expected at night - will be set in _handle_nighttime)"
        )

        # -- Time features -----------------------------------------------------
        df['hour']       = df.index.hour + df.index.minute / 60.0
        df['day_of_year'] = df.index.dayofyear

        hours_per_day  = 24.0
        days_per_year  = 365.25
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / hours_per_day)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / hours_per_day)
        df['day_sin']  = np.sin(2 * np.pi * df['day_of_year'] / days_per_year)
        df['day_cos']  = np.cos(2 * np.pi * df['day_of_year'] / days_per_year)

        # Guard: cyclic features must be in [-1, 1]
        for cyc_col in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']:
            assert df[cyc_col].between(-1.0 - 1e-6, 1.0 + 1e-6).all(), (
                f"Cyclic feature '{cyc_col}' has values outside [-1, 1]: "
                f"min={df[cyc_col].min():.4f}, max={df[cyc_col].max():.4f}"
            )

        # -- Solar times -------------------------------------------------------
        solar_times      = location.get_sun_rise_set_transit(df.index)
        df['solar_noon'] = solar_times['transit']
        df['sunrise']    = solar_times['sunrise']
        df['sunset']     = solar_times['sunset']

        n_nat = df['solar_noon'].isna().sum()
        if n_nat > 0:
            self.logger.warning(
                f"  {n_nat} NaT values in solar_noon (extreme latitude or pvlib issue) "
                f"- forward-filling"
            )
            df['solar_noon'] = df['solar_noon'].ffill().bfill()

        # Guard: no remaining NaT after fill
        assert df['solar_noon'].notna().all(), (
            "solar_noon still contains NaT after ffill/bfill - "
            "check if entire dataset lacks valid solar noon"
        )

        # -- Time relative to solar noon ---------------------------------------
        df['time_to_solar_noon'] = (df.index - df['solar_noon']).dt.total_seconds() / 3600.0

        # Guard: should be in [-profile_half_window_hours * ~1.5, same] roughly
        t2sn_abs_max = df['time_to_solar_noon'].abs().max()
        if t2sn_abs_max > 14.0:
            self.logger.warning(
                f"  time_to_solar_noon max absolute value = {t2sn_abs_max:.2f} h "
                f"(> 14 h - possible solar_noon fill artefact)"
            )

        # -- Day length --------------------------------------------------------
        df['day_length'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 3600.0

        n_bad_dl = (df['day_length'] < 0).sum()
        if n_bad_dl > 0:
            self.logger.warning(
                f"  {n_bad_dl} records with negative day_length - "
                f"likely a sunset < sunrise artefact at high latitudes"
            )

        # -- Location constants ------------------------------------------------
        df['latitude']  = lat
        df['longitude'] = lon

        self.logger.debug(
            f"  Geometry columns added: solar_zenith_angle, solar_elevation, "
            f"solar_azimuth_angle, extraterrestrial_radiation, air_mass, "
            f"hour/day cyclic features, solar_noon/sunrise/sunset, "
            f"time_to_solar_noon, day_length, latitude, longitude"
        )

        return df

    # =========================================================================
    # STAGE 3: CLEAR-SKY
    # =========================================================================

    def _compute_clear_sky(self, df: pd.DataFrame, station: str) -> pd.DataFrame:
        #
        # Compute clear-sky GHI using pvlib Ineichen model.
        # 
        # Args:
        #     df     : DataFrame with solar geometry
        #     station: Station code
        # 
        # Returns:
        #     DataFrame with ghi_clear_sky column added
        #
        metadata = self.station_metadata[station]
        location = pvlib.location.Location(
            latitude=metadata['lat'],
            longitude=metadata['lon'],
            altitude=metadata['elev'],
            tz='UTC'
        )

        try:
            clearsky = location.get_clearsky(df.index, model='ineichen')
            df['ghi_clear_sky'] = clearsky['ghi']
        except Exception as e:
            raise ValueError(
                f"Error computing Ineichen clear sky for {station}: {e}"
            )

        # Clean up: replace NaN/inf with 0 (pvlib returns NaN at night)
        n_nan_cs  = df['ghi_clear_sky'].isna().sum()
        n_inf_cs  = np.isinf(df['ghi_clear_sky']).sum()
        if n_nan_cs > 0 or n_inf_cs > 0:
            self.logger.debug(
                f"  ghi_clear_sky: {n_nan_cs} NaN + {n_inf_cs} inf -> replaced with 0.0"
            )
        df['ghi_clear_sky'] = df['ghi_clear_sky'].replace([np.inf, -np.inf], 0.0)
        df['ghi_clear_sky'] = df['ghi_clear_sky'].fillna(0.0)

        # Ensure non-negative values
        n_neg_cs = (df['ghi_clear_sky'] < 0).sum()
        if n_neg_cs > 0:
            self.logger.debug(
                f"  ghi_clear_sky: {n_neg_cs} negative values clipped to 0.0"
            )
        df['ghi_clear_sky'] = df['ghi_clear_sky'].clip(lower=0.0)

        # Guard: should be finite everywhere now
        assert not df['ghi_clear_sky'].isna().any(), \
            "ghi_clear_sky still contains NaN after fill"
        assert not np.isinf(df['ghi_clear_sky']).any(), \
            "ghi_clear_sky still contains inf after fill"
        assert (df['ghi_clear_sky'] >= 0).all(), \
            "ghi_clear_sky still contains negative values after clip"

        self.logger.debug(
            f"  ghi_clear_sky range: "
            f"[{df['ghi_clear_sky'].min():.1f}, {df['ghi_clear_sky'].max():.1f}] W/m^2"
        )

        return df

    # =========================================================================
    # STAGE 4: NIGHTTIME HANDLING
    # =========================================================================

    def _handle_nighttime(self, df: pd.DataFrame) -> pd.DataFrame:
        #
        # Identify nighttime periods and force all nighttime values to safe defaults.
        # 
        # This must run BEFORE any division operations (CSI, diffuse fraction)
        # to prevent division by zero or near-zero values.
        # 
        # Nighttime is defined as solar zenith angle exceeding the configured threshold.
        # 
        # Args:
        #     df: DataFrame with solar geometry and irradiance
        # 
        # Returns:
        #     DataFrame with nighttime values properly set
        #
        nighttime_mask = df['solar_zenith_angle'] > self.nighttime_zenith_threshold

        n_night = nighttime_mask.sum()
        n_total = len(df)
        n_day   = n_total - n_night
        self.logger.info(
            f"  Nighttime records : {n_night:,}/{n_total:,} "
            f"({100*n_night/max(n_total,1):.1f}%)"
        )
        self.logger.info(
            f"  Daytime  records  : {n_day:,}/{n_total:,} "
            f"({100*n_day/max(n_total,1):.1f}%)"
        )

        # Guard: nighttime fraction should be plausible (roughly 30-70% annually)
        night_frac = n_night / max(n_total, 1)
        if night_frac < 0.20 or night_frac > 0.80:
            self.logger.warning(
                f"  Unusual nighttime fraction: {night_frac:.2%} "
                f"(expected ~30-70% for mid-latitude annual data)"
            )

        # Force all nighttime irradiance to exactly zero
        for col in ['ghi', 'dif', 'dni', 'ghi_clear_sky']:
            n_nonzero_night = (df.loc[nighttime_mask, col] != 0).sum()
            if n_nonzero_night > 0:
                self.logger.debug(
                    f"  Zeroing {n_nonzero_night} non-zero nighttime values in '{col}'"
                )
            df.loc[nighttime_mask, col] = 0.0

        # Set air mass to horizon value at night
        nighttime_am_value = self.air_mass_config['nighttime_value']
        df.loc[nighttime_mask, 'air_mass'] = nighttime_am_value
        self.logger.debug(
            f"  Nighttime air_mass set to configured value: {nighttime_am_value}"
        )

        # Fill any remaining NaN in air mass (daytime NaN from pvlib edge cases)
        n_nan_am_remain = df['air_mass'].isna().sum()
        if n_nan_am_remain > 0:
            self.logger.debug(
                f"  {n_nan_am_remain} remaining NaN in air_mass (daytime edge cases) "
                f"- filling with nighttime_value={nighttime_am_value}"
            )
        df['air_mass'] = df['air_mass'].fillna(nighttime_am_value)

        # Clip to valid range
        am_min = self.air_mass_config['min_value']
        am_max = self.air_mass_config['max_value']
        n_clipped_am = ((df['air_mass'] < am_min) | (df['air_mass'] > am_max)).sum()
        if n_clipped_am > 0:
            self.logger.debug(
                f"  {n_clipped_am} air_mass values clipped to [{am_min}, {am_max}]"
            )
        df['air_mass'] = df['air_mass'].clip(am_min, am_max)

        # Guard: air_mass must be finite and within range everywhere
        assert df['air_mass'].notna().all(), \
            "air_mass still contains NaN after _handle_nighttime"
        assert not np.isinf(df['air_mass']).any(), \
            "air_mass still contains inf after _handle_nighttime"
        assert (df['air_mass'] >= am_min).all() and (df['air_mass'] <= am_max).all(), \
            f"air_mass out of [{am_min}, {am_max}] after clip"

        # Guard: nighttime irradiance must all be exactly 0.0
        for col in ['ghi', 'dif', 'dni', 'ghi_clear_sky']:
            assert (df.loc[nighttime_mask, col] == 0.0).all(), (
                f"Nighttime zeroing failed for '{col}' - non-zero values remain"
            )

        return df

    # =========================================================================
    # STAGE 5: QUALITY CONTROL
    # =========================================================================

    def _apply_quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        #
        # Apply quality control filters to remove erroneous measurements.
        # 
        # Filters applied:
        #   1. Remove negative irradiance during daytime (sensor errors)
        #   2. Remove physically impossible high values
        #   3. Remove cases where DIF exceeds GHI (with tolerance for measurement error)
        # 
        # Args:
        #     df: DataFrame to filter
        # 
        # Returns:
        #     Filtered DataFrame with bad records removed
        #
        initial_count = len(df)
        daytime_mask  = df['solar_zenith_angle'] <= self.daytime_zenith_threshold

        n_daytime = daytime_mask.sum()
        self.logger.debug(
            f"  QC: {n_daytime:,} daytime records ({initial_count:,} total) - "
            f"filters applied to daytime only"
        )

        # Filter 1: Negative irradiance during daytime
        negative_mask = (df['ghi'] < 0) | (df['dif'] < 0)
        bad_negative  = daytime_mask & negative_mask
        n_bad_neg     = bad_negative.sum()
        if n_bad_neg > 0:
            self.logger.debug(
                f"  QC filter 1 (negative irradiance)    : {n_bad_neg:,} records flagged"
            )

        # Filter 2: Physically impossible high values
        bad_ghi = daytime_mask & (df['ghi'] > self.qc_config['max_ghi'])
        bad_dif = daytime_mask & (df['dif'] > self.qc_config['max_dif'])
        n_bad_ghi_high = bad_ghi.sum()
        n_bad_dif_high = bad_dif.sum()
        if n_bad_ghi_high > 0:
            self.logger.debug(
                f"  QC filter 2a (GHI > {self.qc_config['max_ghi']} W/m^2)  : "
                f"{n_bad_ghi_high:,} records flagged"
            )
        if n_bad_dif_high > 0:
            self.logger.debug(
                f"  QC filter 2b (DIF > {self.qc_config['max_dif']} W/m^2)  : "
                f"{n_bad_dif_high:,} records flagged"
            )

        # Filter 3: DIF cannot exceed GHI (with tolerance for measurement error)
        dif_exceeds_ghi = df['dif'] > (df['ghi'] + self.dif_ghi_tolerance)
        bad_dif_ghi     = daytime_mask & dif_exceeds_ghi
        n_bad_dif_ghi   = bad_dif_ghi.sum()
        if n_bad_dif_ghi > 0:
            self.logger.debug(
                f"  QC filter 3 (DIF > GHI + {self.dif_ghi_tolerance} W/m^2): "
                f"{n_bad_dif_ghi:,} records flagged"
            )

        # Combine all bad record masks
        bad_records   = bad_negative | bad_ghi | bad_dif | bad_dif_ghi
        n_bad_total   = bad_records.sum()
        df            = df[~bad_records]
        filtered_count = initial_count - len(df)

        self.logger.info(
            f"  QC summary: {filtered_count:,} records removed "
            f"({100*filtered_count/max(initial_count,1):.2f}% of {initial_count:,})"
        )

        # Guard: no records should have been added
        assert len(df) <= initial_count, \
            "QC filter increased record count - logic error"

        return df

    # =========================================================================
    # STAGE 6: DERIVED VARIABLES
    # =========================================================================

    def _compute_derived_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        #
        # Compute clear-sky index (CSI) and diffuse fraction with safe division logic.
        # 
        # This runs AFTER _handle_nighttime, ensuring nighttime values are already zero.
        # Division is only performed where denominators exceed safety thresholds.
        # 
        # CSI              = GHI / GHI_clear_sky  (cloudiness measure)
        # Diffuse fraction = DIF / GHI             (scattering measure)
        # 
        # Args:
        #     df: DataFrame with irradiance and clear sky values
        # 
        # Returns:
        #     DataFrame with csi and diffuse_fraction columns added
        #
        # Initialize to safe defaults
        df['csi']               = 0.0
        df['diffuse_fraction']  = 0.0

        nighttime_mask = df['solar_zenith_angle'] > self.nighttime_zenith_threshold

        # -- CSI ---------------------------------------------------------------
        csi_valid_mask  = (~nighttime_mask) & (df['ghi_clear_sky'] > self.min_clear_sky_for_csi)
        n_csi_valid     = csi_valid_mask.sum()
        n_csi_skipped   = (~nighttime_mask).sum() - n_csi_valid

        self.logger.debug(
            f"  CSI computation: {n_csi_valid:,} valid daytime records "
            f"(clear_sky > {self.min_clear_sky_for_csi} W/m^2), "
            f"{n_csi_skipped:,} daytime records skipped (low clear-sky)"
        )

        if n_csi_valid > 0:
            df.loc[csi_valid_mask, 'csi'] = (
                df.loc[csi_valid_mask, 'ghi'] /
                df.loc[csi_valid_mask, 'ghi_clear_sky']
            )

        # -- Diffuse fraction --------------------------------------------------
        df_valid_mask = (~nighttime_mask) & (df['ghi'] > self.min_ghi_for_df)
        n_df_valid    = df_valid_mask.sum()
        n_df_skipped  = (~nighttime_mask).sum() - n_df_valid

        self.logger.debug(
            f"  Diffuse fraction computation: {n_df_valid:,} valid daytime records "
            f"(GHI > {self.min_ghi_for_df} W/m^2), "
            f"{n_df_skipped:,} daytime records skipped (low GHI)"
        )

        if n_df_valid > 0:
            df.loc[df_valid_mask, 'diffuse_fraction'] = (
                df.loc[df_valid_mask, 'dif'] /
                df.loc[df_valid_mask, 'ghi']
            )

        # -- Clip to physically valid ranges -----------------------------------
        csi_min = self.qc_config['min_csi']
        csi_max = self.qc_config['max_csi']
        df_min  = self.qc_config['min_diffuse_fraction']
        df_max  = self.qc_config['max_diffuse_fraction']

        n_csi_clipped  = ((df['csi'] < csi_min) | (df['csi'] > csi_max)).sum()
        n_df_clipped   = (
            (df['diffuse_fraction'] < df_min) |
            (df['diffuse_fraction'] > df_max)
        ).sum()

        if n_csi_clipped > 0:
            self.logger.debug(
                f"  CSI: {n_csi_clipped:,} values clipped to [{csi_min}, {csi_max}]"
            )
        if n_df_clipped > 0:
            self.logger.debug(
                f"  Diffuse fraction: {n_df_clipped:,} values clipped to [{df_min}, {df_max}]"
            )

        df['csi']               = df['csi'].clip(csi_min, csi_max)
        df['diffuse_fraction']  = df['diffuse_fraction'].clip(df_min, df_max)

        # Final safety: replace any remaining NaN/inf
        for col in ['csi', 'diffuse_fraction']:
            n_inf = np.isinf(df[col]).sum()
            n_nan = df[col].isna().sum()
            if n_inf > 0 or n_nan > 0:
                self.logger.warning(
                    f"  '{col}' had {n_nan} NaN + {n_inf} inf after clip - "
                    f"replacing with 0.0"
                )
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Guard: final state must be fully clean
        for col in ['csi', 'diffuse_fraction']:
            assert not df[col].isna().any(), \
                f"'{col}' still contains NaN after _compute_derived_variables"
            assert not np.isinf(df[col]).any(), \
                f"'{col}' still contains inf after _compute_derived_variables"

        self.logger.debug(
            f"  CSI range              : [{df['csi'].min():.4f}, {df['csi'].max():.4f}]"
        )
        self.logger.debug(
            f"  Diffuse fraction range : "
            f"[{df['diffuse_fraction'].min():.4f}, {df['diffuse_fraction'].max():.4f}]"
        )

        return df

    # =========================================================================
    # STAGE 7: RESAMPLE
    # =========================================================================

    def _resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        #
        # Resample data to target temporal resolution with appropriate aggregation.
        # 
        # Different variables require different aggregation methods:
        #   - Irradiance : mean (energy is averaged)
        #   - Geometry   : mean (angles are averaged)
        #   - Time markers: first (take start of interval)
        # 
        # applied here:
        #   pandas Resample.mean() does NOT support the min_count keyword argument
        #   (only GroupBy.mean() does).  The equivalent behaviour - producing NaN
        #   for bins with zero raw records rather than a spurious 0.0 - is achieved
        #   by computing bin counts separately and masking where count == 0.
        # 
        # Args:
        #     df: DataFrame at original resolution
        # 
        # Returns:
        #     DataFrame resampled to target resolution
        #
        target_res = self.data_config['target_resolution_minutes']

        # Define column groups by aggregation method.
        irradiance_cols = ['ghi', 'dif', 'dni', 'csi', 'diffuse_fraction', 'ghi_clear_sky']
        geometry_cols = [
            'solar_zenith_angle', 'solar_elevation', 'solar_azimuth_angle',
            'extraterrestrial_radiation', 'time_to_solar_noon',
            'day_length', 'hour', 'day_of_year',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'latitude', 'longitude',
        ]
        air_mass_cols = ['air_mass']   # averaged like geometry, but never forward-filled
        time_cols     = ['solar_noon', 'sunrise', 'sunset']

        # Guard: all expected columns must be present before resampling
        all_expected = (
            [c for c in irradiance_cols if c in df.columns] +
            [c for c in geometry_cols   if c in df.columns] +
            [c for c in air_mass_cols   if c in df.columns]
        )
        missing_before = [c for c in irradiance_cols + geometry_cols + air_mass_cols
                          if c not in df.columns]
        if missing_before:
            self.logger.warning(
                f"  Columns expected for resampling but not found: {missing_before}"
            )

        # -- Build resampled index ---------------------------------------------
        df_resampled = pd.DataFrame(
            index=df.resample(f'{target_res}min').mean().index
        )
        self.logger.debug(
            f"  Resampled index: {len(df_resampled):,} bins "
            f"at {target_res}-min resolution"
        )

        # -- FIX: bin-count guard replaces unsupported min_count=1 -------------
        #
        # Goal: a resampled bin that has ZERO original records must be NaN (honest
        # gap), not 0.0 (false zero that would be treated as real measurement).
        #
        # Why the original code broke:
        #   pandas Resample.mean() does not accept `min_count` as a kwarg.
        #   Only GroupBy.mean() supports it.  Passing it raises:
        #       UnsupportedFunctionCall: numpy operations are not valid with resample.
        #
        # Fix:
        #   1. Compute bin counts once using .count() on a reference column
        #      (count() is NaN-aware: counts non-NaN values per bin; a bin with
        #       no records at all will have count == 0).
        #   2. Call .mean() without kwargs (returns NaN for all-NaN bins already
        #      in modern pandas, but this is NOT guaranteed to return NaN for
        #      zero-record bins in all versions).
        #   3. Explicitly mask bins where count == 0 to NaN (version-safe).
        #
        # This is semantically identical to the intended min_count=1 behaviour.

        ref_col    = next((c for c in irradiance_cols if c in df.columns), None)
        bin_counts = (
            df[ref_col].resample(f'{target_res}min').count()
            if ref_col is not None else None
        )

        if bin_counts is not None:
            n_empty_bins = (bin_counts == 0).sum()
            if n_empty_bins > 0:
                self.logger.debug(
                    f"  {n_empty_bins} bins have zero raw records - "
                    f"will be forced to NaN (honest gap)"
                )

        for col in irradiance_cols + geometry_cols + air_mass_cols:
            if col not in df.columns:
                continue

            # Step 1: mean (NaN for all-NaN bins in pandas >= 1.1)
            df_resampled[col] = df[col].resample(f'{target_res}min').mean()

            # Step 2: force zero-record bins to NaN (version-safe guard)
            if bin_counts is not None:
                df_resampled[col] = df_resampled[col].where(bin_counts > 0, other=np.nan)

        # Time columns: first value in interval
        for col in time_cols:
            if col in df.columns:
                df_resampled[col] = df[col].resample(f'{target_res}min').first()

        # -- Geometry: forward-fill gaps from data outages --------------------
        geom_nan_cols: Dict[str, int] = {}
        for col in geometry_cols:
            if col not in df_resampled.columns:
                continue
            n_nan = int(df_resampled[col].isna().sum())
            if n_nan > 0:
                geom_nan_cols[col] = n_nan
            df_resampled[col] = df_resampled[col].ffill().bfill()

        if geom_nan_cols:
            total_geom_nan = sum(geom_nan_cols.values())
            worst          = max(geom_nan_cols, key=geom_nan_cols.get)
            self.logger.debug(
                f"  Geometry resample gaps: {total_geom_nan} NaN across "
                f"{len(geom_nan_cols)} cols (worst: '{worst}'={geom_nan_cols[worst]}) "
                f"- ffilled from data outage (safe: geometry is deterministic pvlib)"
            )
        else:
            self.logger.debug("  Geometry: no NaN gaps after resampling")

        # Guard: geometry must be fully filled now
        for col in geometry_cols:
            if col in df_resampled.columns:
                n_remaining = df_resampled[col].isna().sum()
                assert n_remaining == 0, (
                    f"Geometry column '{col}' still has {n_remaining} NaN "
                    f"after ffill/bfill - check for completely empty date ranges"
                )

        # -- air_mass: fill gaps with nighttime value --------------------------
        if 'air_mass' in df_resampled.columns:
            n_nan_am = int(df_resampled['air_mass'].isna().sum())
            if n_nan_am > 0:
                self.logger.debug(
                    f"  {n_nan_am} NaN in air_mass after resample - filling with "
                    f"nighttime_value={self.air_mass_config['nighttime_value']} (data gap)"
                )
                df_resampled['air_mass'] = df_resampled['air_mass'].fillna(
                    self.air_mass_config['nighttime_value']
                )
            df_resampled['air_mass'] = df_resampled['air_mass'].clip(
                self.air_mass_config['min_value'],
                self.air_mass_config['max_value']
            )
            # Guard: air_mass must be clean
            assert df_resampled['air_mass'].notna().all(), \
                "air_mass has NaN after resample fill"
            assert not np.isinf(df_resampled['air_mass']).any(), \
                "air_mass has inf after resample"

        # -- Irradiance gap handling: interpolate small, drop large ------------
        max_interp_gap = int(
            self.data_config.get('max_interpolation_gap_minutes', 120) / target_res
        )
        min_present_frac = float(
            self.data_config.get('min_present_fraction', 0.70)
        )

        assert max_interp_gap >= 1, (
            f"max_interpolation_gap_minutes must be >= target_resolution_minutes "
            f"({target_res} min), got "
            f"{self.data_config.get('max_interpolation_gap_minutes')} min"
        )
        assert 0.0 < min_present_frac <= 1.0, (
            f"min_present_fraction must be in (0, 1], got {min_present_frac}"
        )

        self.logger.debug(
            f"  Irradiance gap policy: interpolate <= {max_interp_gap * target_res} min gaps, "
            f"drop days with present fraction < {min_present_frac:.0%}"
        )

        interp_cols            = ['ghi', 'dif', 'dni', 'csi', 'diffuse_fraction', 'ghi_clear_sky']
        n_interpolated_total   = 0
        n_large_gap_total      = 0
        per_col_interp_counts: Dict[str, int] = {}

        for col in interp_cols:
            if col not in df_resampled.columns:
                continue
            n_before = int(df_resampled[col].isna().sum())
            if n_before == 0:
                per_col_interp_counts[col] = 0
                continue

            df_resampled[col] = df_resampled[col].interpolate(
                method='linear',
                limit=max_interp_gap,
                limit_direction='both',
            )
            n_after = int(df_resampled[col].isna().sum())
            n_filled = n_before - n_after
            n_interpolated_total += n_filled
            n_large_gap_total    += n_after
            per_col_interp_counts[col] = n_filled

            if n_filled > 0:
                self.logger.debug(
                    f"    '{col}': {n_filled} intervals interpolated, "
                    f"{n_after} unresolvable (gap too large)"
                )

        # Per-day completeness check
        n_bad_days = 0
        if 'ghi' in df_resampled.columns:
            df_resampled['_date'] = df_resampled.index.date
            daily_present = df_resampled.groupby('_date')['ghi'].apply(
                lambda x: x.notna().mean()
            )
            bad_days = daily_present[daily_present < min_present_frac].index
            n_bad_days = len(bad_days)

            if n_bad_days > 0:
                bad_mask = df_resampled['_date'].isin(bad_days)
                for col in interp_cols:
                    if col in df_resampled.columns:
                        df_resampled.loc[bad_mask, col] = np.nan
                self.logger.debug(
                    f"  {n_bad_days} day(s) marked bad: present fraction < "
                    f"{min_present_frac:.0%} - profiles for these days will be dropped"
                )
                if n_bad_days <= 10:
                    self.logger.debug(
                        f"  Bad days: {sorted(str(d) for d in bad_days)}"
                    )
            else:
                self.logger.debug(
                    f"  All days meet minimum present-fraction threshold "
                    f"({min_present_frac:.0%})"
                )
            df_resampled.drop(columns=['_date'], inplace=True)

        # Clip valid irradiance to non-negative (float noise)
        for col in ['ghi', 'dif', 'dni', 'ghi_clear_sky']:
            if col in df_resampled.columns:
                n_neg = (df_resampled[col] < 0).sum()
                if n_neg > 0:
                    self.logger.debug(
                        f"  '{col}': {n_neg} slightly negative values clipped to 0 "
                        f"(float noise)"
                    )
                df_resampled[col] = df_resampled[col].clip(lower=0.0)

        # Summary
        if n_interpolated_total > 0 or n_large_gap_total > 0 or n_bad_days > 0:
            self.logger.info(
                f"  Resample gap summary: "
                f"{n_interpolated_total} intervals interpolated "
                f"(gap <= {max_interp_gap * target_res} min), "
                f"{n_large_gap_total} intervals unresolvable (gap too large), "
                f"{n_bad_days} entire day(s) invalidated "
                f"(present fraction < {min_present_frac:.0%})"
            )
        else:
            self.logger.info("  Resample: no gaps - dataset is complete")

        return df_resampled

    # =========================================================================
    # STAGE 8: SOLAR-NOON-CENTERED PROFILES
    # =========================================================================

    def _create_solar_noon_centered_profiles(
        self,
        df: pd.DataFrame,
        station: str,
    ) -> List[Dict]:
        #
        # Create profiles centered on solar noon for temporal consistency.
        # 
        # Solar noon centering ensures that:
        #   1. The same timestep index always represents the same solar time
        #   2. Daily variability is aligned across different days and stations
        #   3. The model learns consistent temporal patterns
        # 
        # Each profile spans [solar_noon - half_window, solar_noon + half_window]
        # and is validated for completeness and data quality.
        # 
        # Args:
        #     df     : DataFrame with all processed data
        #     station: Station code
        # 
        # Returns:
        #     List of daily profile dictionaries
        #
        profiles         = []
        expected_length  = self.data_config['profile_length']
        min_completeness = self.qc_config['min_completeness']

        critical_vars = ['csi', 'diffuse_fraction', 'ghi', 'dif']

        # Group data by date
        df['date']   = df.index.date
        unique_dates = sorted(df['date'].unique())

        self.logger.info(
            f"  Building profiles for {len(unique_dates)} calendar days "
            f"(expected profile length = {expected_length} timesteps)"
        )

        # Track rejection reasons for summary
        reject_counts: Dict[str, int] = {
            'no_records'       : 0,
            'nat_solar_noon'   : 0,
            'low_completeness' : 0,
            'wrong_length'     : 0,
            'noon_misalignment': 0,
            'nan_in_critical'  : 0,
            'inf_in_critical'  : 0,
            'nan_in_derived'   : 0,
            'nan_in_geometry'  : 0,
        }

        for date in unique_dates:
            day_df = df[df['date'] == date]

            if len(day_df) == 0:
                reject_counts['no_records'] += 1
                continue

            # Get solar noon for this day
            solar_noon = day_df['solar_noon'].iloc[0]

            if pd.isna(solar_noon):
                self.logger.warning(f"  Skipping {date}: solar noon is NaT")
                reject_counts['nat_solar_noon'] += 1
                continue

            # Define profile window centered on solar noon
            start_time = solar_noon - pd.Timedelta(hours=self.profile_half_window_hours)
            end_time   = solar_noon + pd.Timedelta(hours=self.profile_half_window_hours)

            # Extract profile (may span midnight boundary)
            profile_df = df[(df.index >= start_time) & (df.index < end_time)].copy()

            # -- NaN diagnostics (ALWAYS log, even for rejected profiles) --------
            nan_summary = {
                col: int(profile_df[col].isna().sum())
                for col in ['ghi', 'dif', 'dni', 'csi', 'diffuse_fraction',
                            'ghi_clear_sky', 'air_mass']
                if col in profile_df.columns
            }
            total_nan = sum(nan_summary.values())

            if total_nan > 0:
                nan_details = ", ".join(
                    f"{col}={n}" for col, n in nan_summary.items() if n > 0
                )
                self.logger.info(
                    f"  [{station}] {date} - {total_nan} NaN across columns: "
                    f"{nan_details} "
                    f"(window length={len(profile_df)}/{expected_length})"
                )
            else:
                self.logger.debug(
                    f"  [{station}] {date} - 0 NaN "
                    f"(window length={len(profile_df)}/{expected_length})"
                )

            # -- Completeness check -------------------------------------------
            if len(profile_df) < expected_length * min_completeness:
                self.logger.debug(
                    f"  Skipping {date}: insufficient completeness "
                    f"({len(profile_df)} < {expected_length * min_completeness:.0f} "
                    f"= {expected_length} * {min_completeness})"
                )
                reject_counts['low_completeness'] += 1
                continue

            # -- Strict length check ------------------------------------------
            if len(profile_df) != expected_length:
                self.logger.warning(
                    f"  Skipping {date}: wrong profile length "
                    f"({len(profile_df)} != {expected_length})"
                )
                reject_counts['wrong_length'] += 1
                continue

            # -- Solar noon alignment -----------------------------------------
            middle_idx        = expected_length // 2
            time_at_middle    = profile_df.index[middle_idx]
            time_diff_seconds = abs((time_at_middle - solar_noon).total_seconds())
            time_diff_minutes = time_diff_seconds / 60.0

            if time_diff_minutes > self.solar_noon_tolerance_minutes:
                self.logger.warning(
                    f"  Skipping {date}: solar noon misalignment "
                    f"({time_diff_minutes:.1f} min off, "
                    f"tolerance = {self.solar_noon_tolerance_minutes} min)"
                )
                reject_counts['noon_misalignment'] += 1
                continue

            # Recompute time_to_solar_noon for exact centering
            profile_df['time_to_solar_noon'] = (
                (profile_df.index - solar_noon).total_seconds() / 3600.0
            )

            # Guard: time_to_solar_noon at midpoint should be near 0
            t2sn_at_mid = abs(profile_df['time_to_solar_noon'].iloc[middle_idx])
            assert t2sn_at_mid < (self.solar_noon_tolerance_minutes / 60.0 + 0.01), (
                f"time_to_solar_noon at midpoint = {t2sn_at_mid:.4f} h "
                f"after realignment for {date}"
            )

            # -- Data quality: no NaN or inf in critical variables ------------
            has_invalid = False

            for var in critical_vars:
                if profile_df[var].isna().any():
                    n_nan_var = int(profile_df[var].isna().sum())
                    self.logger.warning(
                        f"  Skipping {date}: {n_nan_var} NaN in '{var}'"
                    )
                    reject_counts['nan_in_critical'] += 1
                    has_invalid = True
                    break
                if np.isinf(profile_df[var]).any():
                    n_inf_var = int(np.isinf(profile_df[var]).sum())
                    self.logger.warning(
                        f"  Skipping {date}: {n_inf_var} inf in '{var}'"
                    )
                    reject_counts['inf_in_critical'] += 1
                    has_invalid = True
                    break

            if has_invalid:
                continue

            # =========================================================================
            # COMPUTE DERIVED FEATURES
            # =========================================================================
            if self.feature_engineer.enabled:
                temp_profile = {
                    'ghi'              : profile_df['ghi'].values,
                    'dif'              : profile_df['dif'].values,
                    'csi'              : profile_df['csi'].values,
                    'diffuse_fraction' : profile_df['diffuse_fraction'].values,
                }

                # Guard: inputs to feature engineer must be finite
                for var_name, var_arr in temp_profile.items():
                    n_bad = int(np.sum(~np.isfinite(var_arr)))
                    if n_bad > 0:
                        self.logger.warning(
                            f"  Skipping {date}: {n_bad} non-finite values in "
                            f"'{var_name}' before feature engineering"
                        )
                        has_invalid = True
                        break

                if has_invalid:
                    reject_counts['nan_in_derived'] += 1
                    continue

                derived_features = self.feature_engineer.compute_all_features(temp_profile)

                # Validate and attach derived features
                for feat_name, feat_values in derived_features.items():
                    feat_arr = np.asarray(feat_values, dtype=np.float32)
                    n_bad    = int(np.sum(~np.isfinite(feat_arr)))
                    if n_bad > 0:
                        self.logger.warning(
                            f"  Skipping {date}: {n_bad} non-finite values in "
                            f"derived feature '{feat_name}'"
                        )
                        has_invalid = True
                        break

                    # Guard: derived feature must have correct length
                    assert len(feat_arr) == expected_length, (
                        f"Derived feature '{feat_name}' has length {len(feat_arr)}, "
                        f"expected {expected_length} for {date}"
                    )
                    profile_df[feat_name] = feat_arr

                if has_invalid:
                    reject_counts['nan_in_derived'] += 1
                    continue

            # =========================================================================
            # BUILD PROFILE DICTIONARY
            # =========================================================================
            profile: Dict = {
                'date'              : str(date),
                'station'           : station,
                'solar_noon'        : str(solar_noon),
                'timestamps'        : profile_df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'ghi'               : profile_df['ghi'].values.astype(np.float32).tolist(),
                'dif'               : profile_df['dif'].values.astype(np.float32).tolist(),
                'csi'               : profile_df['csi'].values.astype(np.float32).tolist(),
                'diffuse_fraction'  : profile_df['diffuse_fraction'].values.astype(np.float32).tolist(),
                # Observation channels: available from real measurements; NOT required
                # at generation time.
                'air_mass'          : (
                    profile_df['air_mass'].values.astype(np.float32).tolist()
                    if 'air_mass' in profile_df.columns else None
                ),
                'dni'               : (
                    profile_df['dni'].values.astype(np.float32).tolist()
                    if 'dni' in profile_df.columns else None
                ),
                'deterministic_geometry' : {},
                'derived_features'       : {},
            }

            # Guard: profile lists must all have the correct length
            for list_key in ['timestamps', 'ghi', 'dif', 'csi', 'diffuse_fraction']:
                assert len(profile[list_key]) == expected_length, (
                    f"Profile field '{list_key}' has length {len(profile[list_key])}, "
                    f"expected {expected_length} for {date}"
                )

            # -- Deterministic geometry features ------------------------------
            for feat in self.config['data']['deterministic_geometry_features']:
                if feat not in profile_df.columns:
                    self.logger.warning(
                        f"  Skipping {date}: missing deterministic feature '{feat}'"
                    )
                    has_invalid = True
                    break

                feat_arr = profile_df[feat].values.astype(np.float32)

                # Guard: geometry must be finite
                n_bad = int(np.sum(~np.isfinite(feat_arr)))
                if n_bad > 0:
                    self.logger.warning(
                        f"  Skipping {date}: {n_bad} non-finite values in "
                        f"geometry feature '{feat}'"
                    )
                    has_invalid = True
                    break

                assert len(feat_arr) == expected_length, (
                    f"Geometry feature '{feat}' has wrong length {len(feat_arr)} "
                    f"for {date}"
                )
                profile['deterministic_geometry'][feat] = feat_arr.tolist()

            if has_invalid:
                reject_counts['nan_in_geometry'] += 1
                continue

            # -- Derived features ----------------------------------------------
            for feat in self.derived_feature_names:
                if feat not in profile_df.columns:
                    self.logger.warning(
                        f"  Expected derived feature '{feat}' not found for {date}"
                    )
                    has_invalid = True
                    break

                feat_arr = profile_df[feat].values.astype(np.float32)

                # Guard: derived features must be finite
                n_bad = int(np.sum(~np.isfinite(feat_arr)))
                if n_bad > 0:
                    self.logger.warning(
                        f"  Skipping {date}: {n_bad} non-finite values in "
                        f"derived feature '{feat}' (post-attach)"
                    )
                    has_invalid = True
                    break

                assert len(feat_arr) == expected_length, (
                    f"Derived feature '{feat}' has wrong length {len(feat_arr)} "
                    f"for {date}"
                )
                profile['derived_features'][feat] = feat_arr.tolist()

            if has_invalid:
                continue

            profiles.append(profile)

        # -- Rejection summary -------------------------------------------------
        total_rejected = sum(reject_counts.values())
        total_input    = len(unique_dates)
        total_accepted = len(profiles)

        self.logger.info(
            f"  Profile summary for {station}: "
            f"{total_accepted} accepted / {total_input} calendar days "
            f"({100*total_accepted/max(total_input,1):.1f}%)"
        )
        if total_rejected > 0:
            self.logger.info(f"  Rejection breakdown:")
            for reason, count in reject_counts.items():
                if count > 0:
                    self.logger.info(f"    {reason:<22}: {count:>4}")

        # Guard: accepted + rejected must equal total calendar days
        assert total_accepted + total_rejected == total_input, (
            f"Profile accounting error: accepted({total_accepted}) + "
            f"rejected({total_rejected}) = {total_accepted + total_rejected} "
            f"!= {total_input} calendar days"
        )

        return profiles

    # =========================================================================
    # SPLIT LOGIC
    # =========================================================================

    def split_by_year(
        self,
        all_profiles: List[Dict],
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        #
        # Split profiles into train/validation/test sets.
        # 
        # Year-based split when years differ across splits; stratified monthly
        # random split when all splits share the same year(s).
        # 
        # Args:
        #     all_profiles: All daily profiles from all stations/years
        # 
        # Returns:
        #     Tuple of (train_profiles, val_profiles, test_profiles)
        #
        assert len(all_profiles) > 0, \
            "split_by_year called with empty profile list"

        train_start = self.data_config['splits']['train']['start_year']
        train_end   = self.data_config['splits']['train']['end_year']
        val_start   = self.data_config['splits']['val']['start_year']
        val_end     = self.data_config['splits']['val']['end_year']
        test_start  = self.data_config['splits']['test']['start_year']
        test_end    = self.data_config['splits']['test']['end_year']

        assert train_start <= train_end, \
            f"train split: start_year ({train_start}) > end_year ({train_end})"
        assert val_start <= val_end, \
            f"val split: start_year ({val_start}) > end_year ({val_end})"
        assert test_start <= test_end, \
            f"test split: start_year ({test_start}) > end_year ({test_end})"

        all_years = (
            set(range(train_start, train_end + 1)) |
            set(range(val_start,   val_end   + 1)) |
            set(range(test_start,  test_end  + 1))
        )
        same_year_split = len(all_years) == 1

        if same_year_split:
            self.logger.warning(
                f"All splits use the same year(s): {all_years} - "
                f"using stratified monthly random split instead of year-based split"
            )
            return self._random_split(all_profiles)

        # -- Year-based split --------------------------------------------------
        train_profiles: List[Dict] = []
        val_profiles:   List[Dict] = []
        test_profiles:  List[Dict] = []
        unassigned:     List[Dict] = []

        for profile in all_profiles:
            year = int(profile['date'].split('-')[0])
            if   train_start <= year <= train_end : train_profiles.append(profile)
            elif val_start   <= year <= val_end   : val_profiles.append(profile)
            elif test_start  <= year <= test_end  : test_profiles.append(profile)
            else                                  : unassigned.append(profile)

        if unassigned:
            unassigned_years = sorted({int(p['date'].split('-')[0]) for p in unassigned})
            self.logger.warning(
                f"  {len(unassigned)} profiles not assigned to any split "
                f"(years: {unassigned_years}) - these are discarded"
            )

        total_assigned = len(train_profiles) + len(val_profiles) + len(test_profiles)

        self.logger.info(
            f"Year-based split: "
            f"Train={len(train_profiles)} ({train_start}-{train_end}), "
            f"Val={len(val_profiles)} ({val_start}-{val_end}), "
            f"Test={len(test_profiles)} ({test_start}-{test_end})"
        )
        self.logger.info(
            f"  Total assigned: {total_assigned} / {len(all_profiles)} "
            f"({100*total_assigned/len(all_profiles):.1f}%)"
        )

        # Guard: at least train and val must be non-empty
        assert len(train_profiles) > 0, \
            "Year-based split produced empty train set - check split year ranges"
        assert len(val_profiles) > 0, \
            "Year-based split produced empty val set - check split year ranges"
        if len(test_profiles) == 0:
            self.logger.warning(
                "Year-based split produced empty test set - "
                "check test split year ranges"
            )

        return train_profiles, val_profiles, test_profiles

    def _random_split(
        self,
        all_profiles: List[Dict],
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        #
        # Stratified random split by month, preserving seasonal distribution.
        # 
        # Plain shuffle-and-slice is NOT safe for solar data: if the random seed
        # happens to cluster summer profiles into train and winter profiles into
        # val/test, the three splits have completely different irradiance
        # distributions.  Stratification by month guarantees each split sees
        # every season proportionally.
        # 
        # Strategy:
        #   - Group profiles by calendar month (12 strata).
        #   - Within each month, shuffle with a fixed seed and slice by ratio.
        #   - Concatenate per-month slices to form the final splits.
        #   - No month is entirely absent from any split (as long as the month
        #     has >= 3 profiles, which is always true for a full year).
        #
        train_ratio = self.data_config['split_ratios']['train']
        val_ratio   = self.data_config['split_ratios']['val']
        test_ratio  = self.data_config['split_ratios']['test']

        assert train_ratio > 0, "split_ratios.train must be > 0"
        assert val_ratio   > 0, "split_ratios.val must be > 0"
        assert test_ratio  >= 0, "split_ratios.test must be >= 0"

        total = train_ratio + val_ratio + test_ratio
        assert total > 0, "split_ratios must sum to a positive value"
        train_ratio /= total
        val_ratio   /= total
        # test_ratio is the remainder - NOT normalised to avoid accumulating
        # rounding errors; test gets exactly what's left.

        random_seed = self.data_config['random_seed']
        rng = random.Random(random_seed)
        self.logger.info(
            f"  Stratified monthly split with seed={random_seed}, "
            f"ratios: train={train_ratio:.2%}, val={val_ratio:.2%}, "
            f"test={1-train_ratio-val_ratio:.2%}"
        )

        train_profiles: List[Dict] = []
        val_profiles:   List[Dict] = []
        test_profiles:  List[Dict] = []

        # Group by month
        from collections import defaultdict
        monthly: Dict[int, List[Dict]] = defaultdict(list)
        for profile in all_profiles:
            month = int(profile['date'].split('-')[1])
            monthly[month].append(profile)

        months_present = sorted(monthly.keys())
        self.logger.debug(
            f"  Monthly bucket sizes: "
            f"{ {m: len(monthly[m]) for m in months_present} }"
        )

        for month in months_present:
            bucket = monthly[month].copy()
            rng.shuffle(bucket)

            n       = len(bucket)
            n_train = max(1, int(n * train_ratio))
            n_val   = max(1, int(n * val_ratio))
            n_test  = n - n_train - n_val

            # Guard: trim val if over-allocated
            if n_test < 0:
                n_val  = max(0, n_val + n_test)
                n_test = n - n_train - n_val

            # Guarantee at least 1 in test when bucket is large enough
            if n_test == 0 and n >= 3:
                n_val  = max(0, n_val - 1)
                n_test = 1

            assert n_train + n_val + n_test == n, (
                f"Month {month}: split sizes {n_train}+{n_val}+{n_test} != {n}"
            )
            assert n_train >= 1, f"Month {month}: train split is empty"

            self.logger.debug(
                f"  Month {month:02d}: {n} profiles -> "
                f"train={n_train}, val={n_val}, test={n_test}"
            )

            train_profiles.extend(bucket[:n_train])
            val_profiles.extend(  bucket[n_train : n_train + n_val])
            test_profiles.extend( bucket[n_train + n_val :])

        # Final shuffle so months are not contiguous within each split
        rng.shuffle(train_profiles)
        rng.shuffle(val_profiles)
        rng.shuffle(test_profiles)

        n_total        = len(all_profiles)
        achieved_train = len(train_profiles) / n_total
        achieved_val   = len(val_profiles)   / n_total
        achieved_test  = len(test_profiles)  / n_total

        # Guard: total must be conserved
        assert len(train_profiles) + len(val_profiles) + len(test_profiles) == n_total, (
            f"Random split lost/gained profiles: "
            f"{len(train_profiles)}+{len(val_profiles)}+{len(test_profiles)} "
            f"!= {n_total}"
        )

        self.logger.info(
            f"  Stratified monthly split result: "
            f"Train={len(train_profiles)} ({achieved_train:.2%}), "
            f"Val={len(val_profiles)} ({achieved_val:.2%}), "
            f"Test={len(test_profiles)} ({achieved_test:.2%})"
        )
        self.logger.info(
            f"  Monthly coverage - "
            f"train: {sorted({int(p['date'].split('-')[1]) for p in train_profiles})}, "
            f"val: {sorted({int(p['date'].split('-')[1]) for p in val_profiles})}, "
            f"test: {sorted({int(p['date'].split('-')[1]) for p in test_profiles})}"
        )

        return train_profiles, val_profiles, test_profiles
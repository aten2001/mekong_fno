# src/runner.py
import numpy as np, pandas as pd, tensorflow as tf
from datetime import timedelta
from .model_fno import SeasonalFNO1D
from .dataio import today_in_station_tz, doy_no_leap, doy_no_leap_vec, doy_sin_cos_series

# ============ Daily climatology (computed from training years) ============
def build_climatology_from_train_years(data, train_years=(2015, 2022)) -> np.ndarray:
    """
    Build a daily climatology vector clim[doy] (doy in 1..365) using the training years only.

    Args:
        data: list from load_range_data -> [time_idx, x_pos, u, h, ts]
        train_years: (start_year, end_year) inclusive

    Returns:
        float32 vector of length 366 where indices 1..365 are valid (index 0 is padding with 0.0).
    """
    df = pd.DataFrame(data, columns=['time_idx', 'x_pos', 'u', 'h', 'ts'])
    ts = pd.to_datetime(df['ts'])
    mask = (ts.dt.year >= train_years[0]) & (ts.dt.year <= train_years[1])
    df_tr = df.loc[mask].copy()

    # map to DOY  without leap day
    base_year = 2001
    df_tr['doy'] = ts[mask].apply(lambda t: pd.Timestamp(base_year, t.month, t.day).dayofyear).to_numpy()

    clim_series = df_tr.groupby('doy')['h'].mean()  # some DOYs may be missing in the index
    # reindex to 1..365 and interpolate/fill
    full_index = pd.RangeIndex(1, 366)  # 1..365
    clim_full = clim_series.reindex(full_index).interpolate(limit_direction='both').bfill().ffill()

    clim_vec = np.zeros(366, dtype=np.float32)
    clim_vec[1:366] = clim_full.values.astype(np.float32)
    return clim_vec

# - TenYearUnifiedRunner (includes: load_range_data / prepare_sequences_no_season / _norm_inputs /
# train_with_dual_window_val / _denorm_pred_anom / evaluate_all / get_daily_h_df /
# _series_true_pred7_by_idx / _rmse_with_shift / scan_best_shift / phase_vs_amplitude_report /
# and the unified predict* interfaces below)
class TenYearUnifiedRunner:
    def __init__(self, csv_files_path, seq_length=90, pred_length=7):
        self.csv_files_path = csv_files_path
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.station_info = {'code': 14501, 'name': 'Stung Treng', 'position': 0}

        # Split: train (2015–2022) / val (2023) / test (2024)
        self.train_years = (2015, 2022)
        self.val_year = 2023
        self.test_year = 2024

        # Two evaluation windows: 60 days each
        self.dry_start_month_day = (2, 1)  # 2/1 – (60 days in non-leap counting)
        self.wet_start_month_day = (8, 1)  # 8/1 – (60 days in non-leap counting)

        # Climatology (set via main)
        self.clim = None  # np.ndarray, len=366, indices 1..365 valid

    def set_climatology(self, clim_vec: np.ndarray):
        """Attach the climatology vector (len=366)."""
        assert isinstance(clim_vec, np.ndarray) and clim_vec.shape[0] == 366
        self.clim = clim_vec.astype(np.float32)

    # ---------- Load ~10 years of data with leap day removed ----------
    def load_range_data(self, start_year=2015, end_year=2024,
                        allow_missing_u=False, u_fill_strategy="ffill_then_mean"):
        """
        Load daily water level (and optional discharge) for a year range.
        Reads MRC CSVs, aggregates to daily means, removes Feb-29, and
        constructs chronological rows with placeholders.
        If ``allow_missing_u=True``, fills ``u`` by forward-fill then mean.

        Args:
          start_year: Inclusive start year.
          end_year: Inclusive end year.
          allow_missing_u: Whether to tolerate missing discharge and fill it.
          u_fill_strategy: Currently informational.

        Returns:
          list[list]: Rows ``[time_idx, x_pos, u, h, ts]`` in ascending date order.
        """
        print(f"Loading station {self.station_info['name']} data for {start_year}-{end_year} ...")
        wl_file = f"{self.csv_files_path}/Water Level.ManualKH_{self.station_info['code']:06d}_{self.station_info['name']}.csv"
        q_file = f"{self.csv_files_path}/Discharge.Calculated daily dischargeKH_{self.station_info['code']:06d}_{self.station_info['name']}.csv"

        # Read raw CSVs
        water_df = pd.read_csv(wl_file)
        water_df['Timestamp (UTC+07:00)'] = pd.to_datetime(water_df['Timestamp (UTC+07:00)'])
        water_df['Date'] = water_df['Timestamp (UTC+07:00)'].dt.date
        water_daily = water_df.groupby('Date')['Value'].mean()

        # Optional discharge; keep permissive
        discharge_daily = None
        try:
            discharge_df = pd.read_csv(q_file)
            discharge_df['Timestamp (UTC+07:00)'] = pd.to_datetime(discharge_df['Timestamp (UTC+07:00)'])
            discharge_df['Date'] = discharge_df['Timestamp (UTC+07:00)'].dt.date
            discharge_daily = discharge_df.groupby('Date')['Value'].mean()
        except Exception:
            if not allow_missing_u:
                raise

        # Build full date range with leap day removed
        start_date = pd.to_datetime(f'{start_year}-01-01')
        end_date = pd.to_datetime(f'{end_year}-12-31')
        dates = pd.date_range(start_date, end_date, freq='D')
        dates = dates[~((dates.month == 2) & (dates.day == 29))]

        rows = []
        for i, ts in enumerate(dates):
            d = ts.date()
            # if water level is missing -> skip (h is required)
            if d not in water_daily.index:
                continue

            h = float(water_daily[d])
            u = np.nan

            # if discharge is available, compute u by the original formula; else handle by allow_missing_u
            if (discharge_daily is not None) and (d in discharge_daily.index):
                Q = float(discharge_daily[d])
                u = Q / (800.0 * max(h, 1e-6) * 0.85)
            else:
                if not allow_missing_u:
                    continue  # if u is required for training and missing -> drop the day

            rows.append([i, 0.0, u, h, ts])

        if allow_missing_u and len(rows) > 0:
            df = pd.DataFrame(rows, columns=['time_idx', 'x_pos', 'u', 'h', 'ts'])
            # u is not used as an input channel; still fill to keep structure consistent
            df['u'] = df['u'].ffill()
            if df['u'].isna().any():
                fillv = float(np.nanmean(df['u'].values))
                df['u'] = df['u'].fillna(fillv)
            data = df[['time_idx', 'x_pos', 'u', 'h', 'ts']].values.tolist()
        else:
            data = rows

        # Print range
        if len(data) > 0:
            print(f"Usable days: {len(data)}")
            print(
                f"Date span: {pd.Timestamp(data[0][4]).strftime('%Y-%m-%d')} → {pd.Timestamp(data[-1][4]).strftime('%Y-%m-%d')}")
        else:
            print("Usable days: 0")
        return data

    # ---------- Helpers ----------
    @staticmethod
    def season_flag_from_month(month):
        """
        Map month to season flag.

        Args:
          month: Month as integer 1..12.

        Returns:
          int: 1 for wet season (May–Oct), else 0 for dry.
        """
        return 1 if 5 <= month <= 10 else 0  # 1=wet, 0=dry

    def get_target_date(self, start_idx, data):
        """
        Compute the target date for a window starting at start_idx.

        Args:
          start_idx: Index of the first day of the input window in ``data``.
          data: Same structure as returned by :meth:`load_range_data`.

        Returns:
          pandas.Timestamp: Target date = window end + ``pred_length`` days.
        """
        target_idx = start_idx + self.seq_length + self.pred_length - 1
        if target_idx < len(data):
            return data[target_idx][4]
        last_date = data[start_idx + self.seq_length - 1][4]
        return last_date + timedelta(days=self.pred_length)

    def prepare_sequences_no_season(self, data):
        """
        Build input tensors (6 channels) and 7-step absolute targets.
        Targets are absolute water levels for next 7 days; anomaly conversion is
        applied later in training.

        Args:
          data: Rows ``[time_idx, x_pos, u, h, ts]`` in ascending date order.

        Returns:
          Tuple:
            - X (np.ndarray): shape (N, L, 6), ``float32``.
            - y (np.ndarray): shape (N, 7, 1), absolute ``float32``.
            - tgt_seasons (np.ndarray): shape (N,), 1=wet else 0.
            - tgt_dates (np.ndarray of pandas.Timestamp): shape (N,), target days.
        """
        X, y, tgt_dates = [], [], []

        # raw columns: [time_idx, x_pos, u, h, ts]
        numeric = np.array([[d[0], d[1], d[2], d[3]] for d in data], dtype=np.float32)
        timestamps = np.array([d[4] for d in data])
        h_all = numeric[:, 3].astype(np.float32)

        # 1-day slope (first day padded with 0)
        dh1 = np.concatenate([[0.0], np.diff(h_all)]).astype(np.float32)

        # seasonal encoding via DOY sin/cos (leap day removed)
        doy_sin, doy_cos = doy_sin_cos_series(timestamps)  # [N], [N]

        # 6-channel features
        feats = np.stack(
            [numeric[:, 0],  # time_idx
             numeric[:, 1],  # x_pos (placeholder)
             h_all,  # h
             dh1,  # dh1
             doy_sin,  # sin(2π·DOY/365)
             doy_cos],  # cos(2π·DOY/365)
            axis=1
        ).astype(np.float32)

        for i in range(len(data) - self.seq_length - self.pred_length + 1):
            base = feats[i:i + self.seq_length, :]  # [L, 6]
            X.append(base)

            future_h = h_all[i + self.seq_length: i + self.seq_length + self.pred_length]
            y.append(future_h.reshape(-1, 1))  # [7,1]

            tgt_date = self.get_target_date(i, data)  # window end + H
            tgt_dates.append(pd.to_datetime(tgt_date).normalize())

        tgt_dates = np.array(tgt_dates)
        tgt_seasons = np.array([1 if 5 <= d.month <= 10 else 0 for d in tgt_dates], np.int32)
        return np.array(X, np.float32), np.array(y, np.float32), tgt_seasons, tgt_dates

    # ---------- Indexing by date range ----------
    @staticmethod
    def _mask_by_date_range(tgt_dates, start_date_str, length_days):
        """
        Boolean mask for a contiguous date window.

        Args:
          tgt_dates: Datetime-like array (normalized).
          start_date_str: ISO string YYYY-MM-DD for window start.
          length_days: Window length in days.

        Returns:
          np.ndarray: Boolean mask of same length as ``tgt_dates``.
        """
        start = pd.to_datetime(start_date_str).normalize()
        end = start + pd.Timedelta(days=length_days - 1)
        return (tgt_dates >= start) & (tgt_dates <= end)

    def _year_window_indices(self, tgt_dates, year, m1, d1, length_days):
        """
        Indices for a year-specific window.

        Args:
          tgt_dates: Datetime-like array (normalized).
          year: Integer target year.
          m1: Start month.
          d1: Start day.
          length_days: Window length.

        Returns:
          np.ndarray: Indices (int) within the requested window.
        """
        start_str = f"{year}-{int(m1):02d}-{int(d1):02d}"
        mask = self._mask_by_date_range(tgt_dates, start_str, length_days)
        return np.where(mask)[0]

    # ---------- Normalization (only time_idx, h, dh1; DOY sin/cos already in [-1,1]) ----------
    @staticmethod
    def _norm_inputs(X, t_mean, t_std, h_in_mean, h_in_std, dh_in_mean, dh_in_std):
        """
        Normalize channels ch0(time_idx), ch2(h), ch3(dh1).

        Args:
          X: Input array of shape (N, L, C).
          t_mean, t_std: Mean/std for time_idx.
          h_in_mean, h_in_std: Mean/std for input h.
          dh_in_mean, dh_in_std: Mean/std for input dh1.

        Returns:
          np.ndarray: Same shape as ``X`` with normalized ch0/ch2/ch3.
        """
        Xn = X.copy()
        Xn[:, :, 0] = (Xn[:, :, 0] - t_mean) / (t_std + 1e-8)  # time_idx
        Xn[:, :, 2] = (Xn[:, :, 2] - h_in_mean) / (h_in_std + 1e-8)  # input h
        Xn[:, :, 3] = (Xn[:, :, 3] - dh_in_mean) / (dh_in_std + 1e-8)  # input dh1
        # Xn[:, :, 4], Xn[:, :, 5] unchanged
        return Xn

    # ---------- Training + validation (single model; train on anomaly space) ----------
    def train_with_dual_window_val(self, X, y_abs, tgt_seasons, tgt_dates,
                                   epochs=300, batch_size=32,
                                   use_time_decay=True, tau_years=12.0):
        """
        Train FNO on standardized anomaly with dual-window validation.

        Args:
          X: Inputs (N, L, 6), absolute features.
          y_abs: Targets (N, 7, 1), absolute water levels.
          tgt_seasons: (N,) season flags for sampling weights.
          tgt_dates: (N,) pandas-like target dates.
          epochs: Max epochs.
          batch_size: Batch size.
          use_time_decay: Whether to downweight older samples exponentially.
          tau_years: Time-decay parameter in years.

        Returns:
          tf.keras.callbacks.History: Training history with metrics/loss.

        Raises:
          AssertionError: If climatology was not set or norm stats missing.
        """
        assert self.clim is not None, "Please call set_climatology(clim_vec) before training!"

        print("\n=== Build splits: train/val/test with dual seasonal windows ===")
        # train: 2015–2022
        train_mask = (tgt_dates >= pd.to_datetime(f"{self.train_years[0]}-01-01")) & \
                     (tgt_dates <= pd.to_datetime(f"{self.train_years[1]}-12-31"))
        train_idx = np.where(train_mask)[0]

        # val: two windows in 2023
        dry_val_idx = self._year_window_indices(tgt_dates, self.val_year, *self.dry_start_month_day, length_days=60)
        wet_val_idx = self._year_window_indices(tgt_dates, self.val_year, *self.wet_start_month_day, length_days=60)
        val_idx = np.concatenate([dry_val_idx, wet_val_idx])

        # test: two windows in 2024
        dry_tst_idx = self._year_window_indices(tgt_dates, self.test_year, *self.dry_start_month_day, length_days=60)
        wet_tst_idx = self._year_window_indices(tgt_dates, self.test_year, *self.wet_start_month_day, length_days=60)

        print(f"Train samples: {len(train_idx)}")
        print(f"Val samples: dry={len(dry_val_idx)}, wet={len(wet_val_idx)}, total={len(val_idx)}")
        print(f"Test samples: dry={len(dry_tst_idx)}, wet={len(wet_tst_idx)}")

        # slice
        X_tr, y_tr_abs = X[train_idx], y_abs[train_idx]
        X_val, y_val_abs = X[val_idx], y_abs[val_idx]

        # ===== Convert targets from absolute to anomaly (relative to target-day DOY climatology) =====
        tgt_doy_all = doy_no_leap_vec(tgt_dates)  # [N]
        clim_targets_all = self.clim[tgt_doy_all]  # [N]
        clim_tr = clim_targets_all[train_idx][:, None, None]  # [N_tr,1,1] broadcast to 7 steps
        clim_val = clim_targets_all[val_idx][:, None, None]

        y_tr = y_tr_abs - clim_tr  # [N_tr, 7, 1] anomaly
        y_val = y_val_abs - clim_val

        # ===== Normalize inputs using training-set stats (time_idx, h, dh1) and standardize anomaly targets =====
        t_mean = float(np.mean(X_tr[:, :, 0]));
        t_std = float(np.std(X_tr[:, :, 0]) + 1e-8)
        h_in_mean = float(np.mean(X_tr[:, :, 2]));
        h_in_std = float(np.std(X_tr[:, :, 2]) + 1e-8)
        dh_in_mean = float(np.mean(X_tr[:, :, 3]));
        dh_in_std = float(np.std(X_tr[:, :, 3]) + 1e-8)

        # target (anomaly) standardization
        h_mean = float(np.mean(y_tr[:, :, 0]));
        h_std = float(np.std(y_tr[:, :, 0]) + 1e-8)

        self.norm_stats = dict(
            t_mean=t_mean, t_std=t_std,
            h_in_mean=h_in_mean, h_in_std=h_in_std,
            dh_in_mean=dh_in_mean, dh_in_std=dh_in_std,
            h_mean=h_mean, h_std=h_std
        )

        # normalize inputs & standardize targets
        X_tr_n = self._norm_inputs(X_tr, t_mean, t_std, h_in_mean, h_in_std, dh_in_mean, dh_in_std)
        X_val_n = self._norm_inputs(X_val, t_mean, t_std, h_in_mean, h_in_std, dh_in_mean, dh_in_std)

        y_tr_n = (y_tr - h_mean) / h_std  # [N, 7, 1]  anomaly -> standardized
        y_val_n = (y_val - h_mean) / h_std

        # ===== Sample weights: seasonal balancing × (optional) time decay =====
        tgt_season_tr = tgt_seasons[train_idx]
        tgt_season_val = tgt_seasons[val_idx]
        n_wet_tr = max(1, int(np.sum(tgt_season_tr == 1)))
        n_dry_tr = max(1, int(np.sum(tgt_season_tr == 0)))
        base_w_tr = np.where(tgt_season_tr == 1, 0.5 / n_wet_tr, 0.5 / n_dry_tr).astype('float32')

        if use_time_decay:
            years = pd.to_datetime(tgt_dates[train_idx]).year.astype('int32')
            w_time = np.exp(-(2024 - years) / float(tau_years)).astype('float32')
            w_tr = base_w_tr * w_time
        else:
            w_tr = base_w_tr

        # emphasize peak/valley by amplitude (range over next-7-day anomaly)
        amp_tr = (np.max(y_tr, axis=1)[:, 0] - np.min(y_tr, axis=1)[:, 0])  # [N]
        q10, q90 = np.quantile(amp_tr, [0.10, 0.90])
        amp_c = np.clip((amp_tr - q10) / (q90 - q10 + 1e-8), 0.0, 1.0)
        edge_w = 1.0 + 0.6 * amp_c  # 1.0~1.6
        w_tr = (w_tr * edge_w).astype('float32')

        n_wet_v = max(1, int(np.sum(tgt_season_val == 1)))
        n_dry_v = max(1, int(np.sum(tgt_season_val == 0)))
        w_val = np.where(tgt_season_val == 1, 0.5 / n_wet_v, 0.5 / n_dry_v).astype('float32')

        # ===== Step-weighted MSE + correlation term (scheduled) =====
        w_steps = tf.constant(np.linspace(2.0, 1.0, 7), dtype=tf.float32)  # emphasize near steps
        alpha = tf.Variable(0.10, trainable=False, dtype=tf.float32)  # weight for correlation term

        def step_corr_loss(y_true, y_pred):
            # per-sample weighted MSE over 7 steps → [B]
            se = tf.square(y_true - y_pred)[:, :, 0]  # [B,7]
            mse_per_sample = tf.reduce_mean(se * w_steps, axis=1)  # [B]

            # per-sample correlation over 7 steps → [B]
            yt = y_true[:, :, 0]  # [B,7]
            yp = y_pred[:, :, 0]  # [B,7]
            yt = yt - tf.reduce_mean(yt, axis=1, keepdims=True)
            yp = yp - tf.reduce_mean(yp, axis=1, keepdims=True)
            num = tf.reduce_sum(yt * yp, axis=1)  # [B]
            den = tf.sqrt(tf.reduce_sum(yt ** 2, axis=1) * tf.reduce_sum(yp ** 2, axis=1) + 1e-8)
            corr = num / den  # [B]

            return mse_per_sample + alpha * (1.0 - corr)

        class AlphaScheduler(tf.keras.callbacks.Callback):
            """0-10:0.10 → 11-30:linear to 0.20 → 31-39:0.20 → ≥40:back to 0.10"""

            def on_epoch_begin(self, epoch, logs=None):
                if epoch <= 10:
                    a = 0.10
                elif epoch <= 30:
                    a = 0.10 + (epoch - 10) * (0.20 - 0.10) / (30 - 10)
                elif epoch < 40:
                    a = 0.20
                else:
                    a = 0.10
                alpha.assign(a)

        # ===== Model capacity =====
        self.model = SeasonalFNO1D(
            modes=64, width=96, num_layers=4,
            input_features=X.shape[2], dropout_rate=0.1, l2=1e-5
        )
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=step_corr_loss, weighted_metrics=[])

        callbacks = [
            AlphaScheduler(),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-5, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1),
        ]

        print("\nStart training (anomaly target; validation = 2023 dry/wet windows; "
              "input=[time,x_pos,h,dh1,doy_sin,doy_cos], output=7-step h_anom)...")
        self.history = self.model.fit(
            X_tr_n, y_tr_n,
            sample_weight=w_tr,
            validation_data=(X_val_n, y_val_n, w_val),
            epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks
        )

        # Save indices for later evaluation/plotting
        self.idxs = {
            'train': train_idx,
            'val_all': val_idx,
            'val_dry': dry_val_idx,
            'val_wet': wet_val_idx,
            'tst_dry': dry_tst_idx,
            'tst_wet': wet_tst_idx
        }
        # Keep absolute ground truth and target dates (climatology added back during eval)
        self.data_cache = (X, y_abs, tgt_dates)
        return self.history

    # ---------- De-standardize predictions (anomaly space) ----------
    def _denorm_pred_anom(self, y_pred_n):  # y_pred_n: [N, 7, 1] (standardized)
        """
        De-standardize anomaly predictions.

        Args:
          y_pred_n: Standardized anomaly, shape (N, 7, 1).

        Returns:
          np.ndarray: De-standardized anomaly, shape (N, 7, 1).
        """
        return y_pred_n * self.norm_stats['h_std'] + self.norm_stats['h_mean']

    def _eval_by_indices(self, idx, title):
        """
        Evaluate RMSE/MAE for a subset of samples.
        Builds normalized inputs, runs model, converts to absolute by adding
        target-day DOY climatology, and compares Day-7 to truth.

        Args:
          idx: Indices (1-D array-like) to evaluate.
          title: Tag used in printed logs.

        Returns:
          dict: {
            'y_true': np.ndarray[N],
            'y_pred': np.ndarray[N],
            'h_rmse': float,
            'h_mae' : float,
            'n'     : int,
            'title' : str
          }
        """
        X, y_abs, tgt_dates = self.data_cache
        st = self.norm_stats

        # normalize inputs
        Xn = self._norm_inputs(X[idx], st['t_mean'], st['t_std'],
                               st['h_in_mean'], st['h_in_std'],
                               st['dh_in_mean'], st['dh_in_std'])

        # ground truth: 7th day (absolute)
        y_true7 = y_abs[idx][:, -1, 0]  # [N]

        # prediction: standardized anomaly -> anomaly -> add climatology at target-day (7th) DOY
        y_pred_n = self.model.predict(Xn, verbose=0)  # [N, 7, 1] standardized anomaly
        y_pred_anom = self._denorm_pred_anom(y_pred_n)  # de-standardized anomaly
        tgt_doy = doy_no_leap_vec(pd.to_datetime(tgt_dates[idx]).normalize())
        clim_add = self.clim[tgt_doy][:, None, None]  # [N,1,1]
        y_pred_abs = y_pred_anom + clim_add  # [N,7,1]
        y_pred7 = y_pred_abs[:, -1, 0]  # [N]

        # metrics
        h_rmse = float(np.sqrt(np.mean((y_pred7 - y_true7) ** 2)))
        h_mae = float(np.mean(np.abs(y_pred7 - y_true7)))

        print(f"{title} — RMSE={h_rmse:.3f} m, MAE={h_mae:.3f} m  (n={len(idx)})")
        return dict(y_true=y_true7, y_pred=y_pred7,
                    h_rmse=h_rmse, h_mae=h_mae,
                    n=len(idx), title=title)

    def evaluate_all(self):
        """
        Evaluate on validation (2023 dry/wet) and test (2024 dry/wet) windows.
        Computes per-window RMSE/MAE and prints a weighted 2024 RMSE/MAE
        (weight by sample count).

        Returns:
          dict: Keys {'val_dry','val_wet','tst_dry','tst_wet'}.
        """
        print("\n=== Evaluation on validation (2023) and test (2024) windows ===")
        res = {}
        res['val_dry'] = self._eval_by_indices(self.idxs['val_dry'], "Val-dry (2023)")
        res['val_wet'] = self._eval_by_indices(self.idxs['val_wet'], "Val-wet (2023)")
        res['tst_dry'] = self._eval_by_indices(self.idxs['tst_dry'], "Test-dry (2024)")
        res['tst_wet'] = self._eval_by_indices(self.idxs['tst_wet'], "Test-wet (2024)")

        # weighted RMSE (via MSE) and weighted MAE for 2024
        total_n = res['tst_dry']['n'] + res['tst_wet']['n']
        h_mse_w = (
                          res['tst_dry']['n'] * (res['tst_dry']['h_rmse'] ** 2) +
                          res['tst_wet']['n'] * (res['tst_wet']['h_rmse'] ** 2)
                  ) / max(1, total_n)
        h_rmse_w = float(np.sqrt(h_mse_w))

        h_mae_w = (
                          res['tst_dry']['n'] * res['tst_dry']['h_mae'] +
                          res['tst_wet']['n'] * res['tst_wet']['h_mae']
                  ) / max(1, total_n)

        print(f"\nTest — 2024 weighted RMSE/MAE: RMSE={h_rmse_w:.3f} m, MAE={h_mae_w:.3f} m")
        return res

    def get_daily_h_df(self, year):
        """
        Return a daily-aligned DataFrame for a given year.
        Uses the model to produce Day-7 absolute predictions on all target
        dates in ``year``, aligned with the observed Day-7 ground truth.

        Args:
          year: Integer target year.

        Returns:
          pandas.DataFrame | None:
            Columns ``['date','h_true','h_pred']`` sorted
            by date, or ``None`` if the year has no samples.
        """
        X, Y_abs, tgt_dates = self.data_cache
        tgt_dates_ts = pd.to_datetime(tgt_dates).normalize()

        idx = np.flatnonzero(tgt_dates_ts.year == int(year))
        if len(idx) == 0:
            return None

        st = self.norm_stats
        Xn = self._norm_inputs(X[idx], st['t_mean'], st['t_std'],
                               st['h_in_mean'], st['h_in_std'],
                               st['dh_in_mean'], st['dh_in_std'])
        y_pred_n = self.model.predict(Xn, verbose=0)  # [N, 7, 1] standardized anomaly
        y_pred_anom = self._denorm_pred_anom(y_pred_n)  # anomaly
        # add climatology at target-day (7th) DOY
        tgt_doy = doy_no_leap_vec(tgt_dates_ts[idx])
        clim_add = self.clim[tgt_doy][:, None, None]  # [N,1,1]
        y_pred_abs = y_pred_anom + clim_add

        h_pred = y_pred_abs[:, -1, 0]  # get 7th day
        h_true = Y_abs[idx][:, -1, 0]

        df = pd.DataFrame({
            "date": tgt_dates_ts[idx],
            "h_true": h_true,
            "h_pred": h_pred,
        })
        return df.sort_values("date").reset_index(drop=True)

    # ---------- Utilities: series of (date, true-7th, pred-7th) for a set of indices ----------
    def _series_true_pred7_by_idx(self, idx: np.ndarray):
        """
        Return aligned (dates, truth, pred) series for Day-7.

        Args:
          idx: 1-D integer indices to extract.

        Returns:
          Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - dates: numpy datetime64 array in ascending order.
            - y_true7: float32 array of true Day-7 absolute WL.
            - y_pred7: float32 array of predicted Day-7 absolute WL.
        """
        X, y_abs, tgt_dates = self.data_cache
        st = self.norm_stats

        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            return np.array([]), np.array([]), np.array([])

        # normalize inputs
        Xn = self._norm_inputs(X[idx], st['t_mean'], st['t_std'],
                               st['h_in_mean'], st['h_in_std'],
                               st['dh_in_mean'], st['dh_in_std'])

        # predict anomaly -> de-standardize -> add climatology (target-day DOY)
        y_pred_n = self.model.predict(Xn, verbose=0)  # [N, 7, 1]
        y_pred_an = self._denorm_pred_anom(y_pred_n)  # anomaly
        tgt_doy = doy_no_leap_vec(pd.to_datetime(np.asarray(tgt_dates)[idx]).normalize())
        clim_add = self.clim[tgt_doy][:, None, None]  # [N,1,1]
        y_pred_abs = y_pred_an + clim_add  # [N, 7, 1]

        dates = pd.to_datetime(np.asarray(tgt_dates)[idx]).normalize()
        y_true7 = y_abs[idx][:, -1, 0].astype(np.float32)  # [N]
        y_pred7 = y_pred_abs[:, -1, 0].astype(np.float32)  # [N]

        # sort by date
        order = np.argsort(dates.values)
        return dates.values[order], y_true7[order], y_pred7[order]

    # ---------- Compute RMSE under an integer time shift k over the overlapping region ----------
    @staticmethod
    def _rmse_with_shift(y_true: np.ndarray, y_pred: np.ndarray, k: int):
        """
        RMSE under integer phase shift over the overlapping region.

        Args:
          y_true: Ground-truth array (N,).
          y_pred: Prediction array (N,).
          k: Integer shift.

        Returns:
          Tuple[float, int]: (rmse, overlap_length). If overlap < 10, returns (nan, 0).
        """
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        n = len(y_true)
        if n != len(y_pred) or n == 0:
            return np.nan, 0

        if k > 0:
            # Prediction leads: align y_pred[k:] with y_true[:-k]
            yt = y_true[:-k] if k < n else np.array([])
            yp = y_pred[k:] if k < n else np.array([])
        elif k < 0:
            s = -k
            # Prediction lags: align y_pred[:-s] with y_true[s:]
            yt = y_true[s:] if s < n else np.array([])
            yp = y_pred[:-s] if s < n else np.array([])
        else:
            yt = y_true
            yp = y_pred

        m = len(yt)
        if m < 10:
            return np.nan, 0
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        return rmse, m

    # ---------- Grid-search k ∈ [-K, K] to find best shift on a set of indices ----------
    def _scan_best_shift(self, idx: np.ndarray, K: int = 10, tag: str = ""):
        """
        Grid-search best integer phase shift in [-K, K].

        Args:
          idx: Indices to select a contiguous time series.
          K: Maximum absolute shift to scan.
          tag: Label for logging.

        Returns:
          dict: {
            'best_k': int,
            'best_rmse': float,
            'base_rmse': float,
            'gain': float  # base_rmse - best_rmse
          }
        """
        dates, yt, yp = self._series_true_pred7_by_idx(idx)
        if len(yt) == 0:
            print(f"{tag} contains no samples; skip phase scan.")
            return dict(best_k=0, best_rmse=np.nan, base_rmse=np.nan, gain=np.nan)

        base_rmse, _ = self._rmse_with_shift(yt, yp, 0)
        best_k, best_rmse = 0, base_rmse
        for k in range(-K, K + 1):
            rmse_k, m = self._rmse_with_shift(yt, yp, k)
            if np.isnan(rmse_k):
                continue
            if rmse_k < best_rmse:
                best_rmse, best_k = rmse_k, k
        gain = (base_rmse - best_rmse) if (base_rmse is not None and best_rmse is not None) else np.nan
        print(f"{tag} phase scan: base RMSE={base_rmse:.3f}, best k={best_k:+d}, "
              f"aligned RMSE={best_rmse:.3f}, gain={gain:.3f}")
        return dict(best_k=best_k, best_rmse=best_rmse, base_rmse=base_rmse, gain=gain)

    # ---------- Find k* on 2023 val windows, then fix k on 2024 test windows ----------
    def phase_vs_amplitude_report(self, K: int = 10):
        """
        Report phase alignment gains on 2023 windows and apply to 2024.

        Workflow:
        1) Scan k ∈ [-K, K] for 2023 (merged/dry/wet) and record best k, gain.
        2) Apply those k* to 2024 (dry/wet and merged) and report before/after RMSE.
        3) Return the aggregated report dict.

        Args:
          K: Maximum absolute shift to consider.

        Returns:
          dict: {
            'val': {'all'|'dry'|'wet': {'best_k','best_rmse','base_rmse','gain'}},
            'test_applied': {'all'|'dry'|'wet': {'k','rmse_before','rmse_after','gain'}}
          }
        """
        ids = self.idxs  # Saved during the training function
        report = {"val": {}, "test_applied": {}}

        # ---- 2023: merged / dry / wet ----
        scan_all = self._scan_best_shift(ids['val_all'], K, tag="2023-merged")
        scan_dry = self._scan_best_shift(ids['val_dry'], K, tag="2023-dry")
        scan_wet = self._scan_best_shift(ids['val_wet'], K, tag="2023-wet")
        report["val"]["all"] = scan_all
        report["val"]["dry"] = scan_dry
        report["val"]["wet"] = scan_wet

        ## ---- 2024: fix k from 2023 and compute aligned RMSEs ----
        def _apply_k_and_rmse(idx, k, tag):
            dates, yt, yp = self._series_true_pred7_by_idx(idx)
            rmse0, _ = self._rmse_with_shift(yt, yp, 0)
            rmseK, _ = self._rmse_with_shift(yt, yp, k)
            gain = rmse0 - rmseK
            print(f"{tag} apply k={k:+d}: RMSE {rmse0:.3f} → aligned {rmseK:.3f}, gain={gain:.3f}")
            return dict(k=k, rmse_before=rmse0, rmse_after=rmseK, gain=gain)

        # 2024 merged: use k* from 2023-merged
        res_all = _apply_k_and_rmse(np.concatenate([ids['tst_dry'], ids['tst_wet']]),
                                    scan_all["best_k"], "2024-merged")
        # 2024 dry/wet: use respective k* from 2023 dry/wet
        res_dry = _apply_k_and_rmse(ids['tst_dry'], scan_dry["best_k"], "2024-dry")
        res_wet = _apply_k_and_rmse(ids['tst_wet'], scan_wet["best_k"], "2024-wet")

        report["test_applied"]["all"] = res_all
        report["test_applied"]["dry"] = res_dry
        report["test_applied"]["wet"] = res_wet
        return report

    # ========= single-day & 'today' inference interfaces =========
    def _build_window_for_date(self, water_daily: pd.Series,
                               discharge_daily: pd.Series | None,
                               date_anchor: pd.Timestamp):
        """
        Construct a length-L normalized input window ending at ``date_anchor``.
        Collects daily values (skipping Feb 29 when needed), computes
        ``dh1``, time indices, DOY sin/cos, and normalizes channels 0/2/3.

        Args:
          water_daily: Series (index=python date -> float meters).
          discharge_daily: Unused placeholder for structure compatibility.
          date_anchor: Inclusive window end date (normalized inside).

        Returns:
          np.ndarray: Shape (1, L, 6), normalized input features.

        Raises:
          ValueError: If any required day is missing from ``water_daily``.
        """
        L = int(self.seq_length)
        end_date = pd.to_datetime(date_anchor).normalize()
        start_date = end_date - pd.Timedelta(days=L - 1)

        # Collect L non-leap days
        days = pd.date_range(start_date, end_date, freq="D")
        days = days[~((days.month == 2) & (days.day == 29))]
        if len(days) < L:
            d = start_date - pd.Timedelta(days=1)
            buf = []
            while len(days) + len(buf) < L:
                if not (d.month == 2 and d.day == 29):
                    buf.append(d)
                d -= pd.Timedelta(days=1)
            days = pd.DatetimeIndex(list(reversed(buf)) + list(days))

        # Gather daily h
        h_vals = []
        for dt in days:
            key = getattr(dt, "date", lambda: dt)()
            if key not in water_daily.index:
                raise ValueError(f"Missing water_daily value for {dt.date()}")
            h_vals.append(float(water_daily[key]))
        h_vals = np.asarray(h_vals, dtype=np.float32)

        # dh1 / time_idx / x_pos / DOY sin-cos
        dh1 = np.concatenate([[0.0], np.diff(h_vals)]).astype(np.float32)

        def _time_idx_for_date(dt: pd.Timestamp) -> int:
            base = pd.Timestamp(f"{self.train_years[0]}-01-01")
            all_days = pd.date_range(base, dt, freq='D')
            all_days = all_days[~((all_days.month == 2) & (all_days.day == 29))]
            return len(all_days) - 1

        t_idx = np.asarray([_time_idx_for_date(dt) for dt in days], dtype=np.float32)
        x_pos = np.zeros_like(t_idx, dtype=np.float32)
        doy_sin, doy_cos = doy_sin_cos_series(days)

        feats6 = np.stack([t_idx, x_pos, h_vals, dh1, doy_sin, doy_cos], axis=1).astype(np.float32)  # [L,6]

        # Normalize (only ch0/ch2/ch3)
        st = self.norm_stats
        Xn = feats6.copy()[None, :, :]  # [1,L,6]
        Xn[:, :, 0] = (Xn[:, :, 0] - st['t_mean']) / (st['t_std'] + 1e-8)
        Xn[:, :, 2] = (Xn[:, :, 2] - st['h_in_mean']) / (st['h_in_std'] + 1e-8)
        Xn[:, :, 3] = (Xn[:, :, 3] - st['dh_in_mean']) / (st['dh_in_std'] + 1e-8)
        return Xn

    def predict_h_on_date(self, water_daily: pd.Series, discharge_daily: pd.Series | None,
                          date_anchor: str | pd.Timestamp) -> float:
        """
        Predict absolute WL for the target day = date_anchor + pred_length.

        Args:
          water_daily: Daily series (index=python date -> float).
          discharge_daily: Placeholder, not used.
          date_anchor: Anchor date (string or Timestamp).

        Returns:
          float: Predicted absolute water level (meters) for Day-7 relative to anchor.
        """
        date_anchor = pd.to_datetime(date_anchor).normalize()
        _, h_seq = self.predict_h_range(water_daily, discharge_daily, date_anchor, return_dates=True)
        return float(h_seq[-1])

    def predict_today_h(self, water_daily: pd.Series, discharge_daily: pd.Series | None = None) -> float:
        """
        Predict absolute WL for 'today + pred_length' in station timezone (UTC+07).

        Args:
          water_daily: Daily series (index=python date -> float).
          discharge_daily: Placeholder, not used.

        Returns:
          float: Predicted absolute water level (meters) for today+7.
        """
        today = today_in_station_tz()
        return self.predict_h_on_date(water_daily, discharge_daily, today)

    def predict_h_range(self, water_daily: pd.Series, discharge_daily: pd.Series | None = None,
                        date_anchor: str | pd.Timestamp | None = None,
                        return_dates: bool = False, clim_mode: str = "target7",
                        return_anomaly: bool = False):
        """
        Predict the full 7-step absolute WL sequence consistent with training/eval.
        model → standardized anomaly (7) → de-standardize → add climatology.

        Climatology modes:
          * ``"target7"`` (default): Add the DOY climatology of the 7th day to **all** 7 steps.
            (Matches training/evaluation definition used across scripts.)
          * ``"daywise"``: Add per-step DOY climatology (more physical but different from target7 definition).

        Args:
          water_daily: Series (index=python date -> float meters).
          discharge_daily: Placeholder; not used.
          date_anchor: Window end date. If None, uses station "today" (UTC+07).
          return_dates: If True, also returns the 7 future dates.
          clim_mode: Either ``"target7"`` or ``"daywise"``.
          return_anomaly: If True, additionally return the anomaly sequence.

        Returns:
          Union[
            np.ndarray,                                   # (7,)
            Tuple[pd.DatetimeIndex, np.ndarray],          # if return_dates
            Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]  # if return_dates and return_anomaly
          ]

        Raises:
          AssertionError: If climatology or normalization stats are not set.
          ValueError: If required daily values are missing for the input window.
        """
        assert self.clim is not None, "Complete set_climatology(clim_vec)！first"
        assert hasattr(self, "norm_stats"), "Complete training first to obtain norm_stats!"

        if date_anchor is None:
            date_anchor = today_in_station_tz()
        date_anchor = pd.to_datetime(date_anchor).normalize()

        # ------- Build a length-L input window with leap day removed -------
        L = int(self.seq_length)
        # collect L days backward from end (skip Feb 29)
        days = []
        d = date_anchor
        while len(days) < L:
            if not (d.month == 2 and d.day == 29):
                days.append(d)
            d -= pd.Timedelta(days=1)
        days = days[::-1]  # Ascending in time

        # build 6-channel inputs; x_pos=0.0; u not used
        def _time_idx_for_date(dt: pd.Timestamp) -> int:
            base = pd.Timestamp(f"{self.train_years[0]}-01-01")
            all_days = pd.date_range(base, dt, freq='D')
            all_days = all_days[~((all_days.month == 2) & (all_days.day == 29))]
            return len(all_days) - 1

        # build 6-channel inputs; x_pos=0.0; u not used
        h_vals = []
        for dt in days:
            key = getattr(dt, "date", lambda: dt)()  # Handle both date and Timestamp
            if key not in water_daily.index:
                raise ValueError(f"Missing water_daily value for {dt.date()}, cannot build inference window.")
            h_vals.append(float(water_daily[key]))
        h_vals = np.asarray(h_vals, dtype=np.float32)
        dh1 = np.concatenate([[0.0], np.diff(h_vals)]).astype(np.float32)
        t_idx = np.asarray([_time_idx_for_date(dt) for dt in days], dtype=np.float32)
        x_pos = np.zeros_like(t_idx, dtype=np.float32)

        doy_sin, doy_cos = doy_sin_cos_series(days)

        feats6 = np.stack([t_idx, x_pos, h_vals, dh1, doy_sin, doy_cos], axis=1).astype(np.float32)  # [L,6]

        # normalize (ch0/ch2/ch3)
        st = self.norm_stats
        Xn = feats6.copy()[None, :, :]  # [1,L,6]
        Xn[:, :, 0] = (Xn[:, :, 0] - st['t_mean']) / (st['t_std'] + 1e-8)
        Xn[:, :, 2] = (Xn[:, :, 2] - st['h_in_mean']) / (st['h_in_std'] + 1e-8)
        Xn[:, :, 3] = (Xn[:, :, 3] - st['dh_in_mean']) / (st['dh_in_std'] + 1e-8)

        # ------- Forward pass: standardized anomaly -> de-standardize -------
        y_pred_n = self.model.predict(Xn, verbose=0)  # [1,7,1]
        y_pred_anom = (y_pred_n * st['h_std'] + st['h_mean'])[0, :, 0]  # [7]

        # ------- Add back climatology -------
        # future 7 dates
        fut_dates = pd.date_range(date_anchor + pd.Timedelta(days=1), periods=int(self.pred_length), freq='D')
        fut_dates = fut_dates[~((fut_dates.month == 2) & (fut_dates.day == 29))]
        if len(fut_dates) != self.pred_length:
            # extreme edge case across Feb 29: pad until length==7 while skipping Feb 29
            while len(fut_dates) < self.pred_length:
                fut_dates = fut_dates.append(fut_dates[-1:] + pd.Timedelta(days=1))
                fut_dates = fut_dates[~((fut_dates.month == 2) & (fut_dates.day == 29))]

        if clim_mode == "daywise":
            # add per-day DOY climatology
            fut_doy = doy_no_leap_vec(fut_dates)
            clim_add = self.clim[fut_doy].astype(np.float32)  # [7]
        else:
            # default (matches training/eval): add target (7th day) DOY climatology to all steps
            tgt_day = date_anchor + pd.Timedelta(days=int(self.pred_length))
            tgt_doy = doy_no_leap(pd.to_datetime(tgt_day).normalize())
            clim_add = np.full((self.pred_length,), float(self.clim[tgt_doy]), dtype=np.float32)

        y_pred_abs = (y_pred_anom + clim_add).astype(np.float32)  # [7]

        if return_dates and return_anomaly:
            return fut_dates, y_pred_abs, y_pred_anom
        if return_dates:
            return fut_dates, y_pred_abs
        if return_anomaly:
            return y_pred_abs, y_pred_anom
        return y_pred_abs
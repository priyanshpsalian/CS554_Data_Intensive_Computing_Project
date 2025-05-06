
# %%

import os
import time
import math
import copy


import numpy as np
import pandas as pd
pd.Series.iteritems = pd.Series.items
import matplotlib.pyplot as plt
import seaborn as sns
from finta import TA
import sklearn.preprocessing as pp

import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary


# %%

device = torch.device('cpu')

# %%
data = pd.read_csv("../data/btc_with_sentiment.csv")

columns_dict = dict(t='Unix_timestamp',
                    o='Opening_price',
                    h='Highest_price',
                    l='Lowest_price',
                    c='Closing_price',
                    v='Volume_of_transactions')

data.rename(columns=columns_dict, inplace=True)


# %%
data = data.sort_values(by='Unix_timestamp').reset_index(drop=True)

converted_time = pd.to_datetime(data['Unix_timestamp'], unit='ms') + pd.Timedelta(hours=8)
data.insert(loc=data.columns.get_loc('Unix_timestamp'), column='Timestamp', value=converted_time)
data.drop(columns=['Unix_timestamp'], inplace=True)


# %%
def plot_scaled_features(data, features_to_plot, scaler=None, xlabel="Time"):
    fig, axis = plt.subplots(figsize=(16, 8))
    axis.set(title="Features' Values Through Time",
             xlabel=xlabel,
             ylabel="Features' Values [a.u]")
    axis.grid(visible=True)

    features_available = [col for col in features_to_plot if col in data]

    for col in features_available:
        values = data[col].values.reshape(-1, 1)
        processed = scaler.fit_transform(values) if scaler else values
        axis.plot(processed, label=col)

    axis.legend()
    plt.tight_layout()
    plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation(data, title, figsize=(10, 5), cmap='coolwarm'):
    corr = 100 * data.corr()
    corr = corr.iloc[:, ::-1]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', linewidths=5, cmap=cmap)
    ax.set_title(title)
    ax.tick_params(axis='x', labelrotation=60)
    plt.show()


# %%
num_encoder_layers = 4
num_decoder_layers = 4
periodic_features = 10
out_features = 60 
nhead = 15 
dim_feedforward = 384
dropout = 0.0
activation = 'gelu' 

# %%
plot_correlation(data, 
                     title="Correlations")

# %%
def add_finta_feature(data, data_finta, feature_names, both_columns_features):
    for feature_name in feature_names:
        result = getattr(TA, feature_name)(data_finta)

        if feature_name in both_columns_features and hasattr(result, "shape") and result.shape[1] >= 2:
            col1, col2 = result.iloc[:, 0], result.iloc[:, 1]
            data[f"{feature_name}_1"] = col1
            data[f"{feature_name}_2"] = col2
        else:
            first_col = result.iloc[:, 0] if hasattr(result, "shape") and result.ndim > 1 else result
            data[feature_name] = first_col

data_finta = pd.DataFrame(index=data.index)
for col in ['open', 'high', 'low', 'close', 'volume']:
    data_finta[col] = data[[k for k, v in {
        'open': 'Opening_price',
        'high': 'Highest_price',
        'low': 'Lowest_price',
        'close': 'Closing_price',
        'volume': 'Volume_of_transactions'
    }.items() if k == col][0]]

data_min = data.copy()
extra_features = ['TRIX', 'VWAP', 'MACD', 'EV_MACD', 'MOM', 'RSI', 'IFT_RSI', 'TR', 'ATR', 'BBWIDTH', 'DMI', 'ADX', 'STOCHRSI',
                  'MI', 'CHAIKIN', 'VZO', 'PZO', 'EFI', 'EBBP', 'BASP', 'BASPN', 'WTO', 'SQZMI', 'VFI', 'STC']
both_columns_features = ["DMI", "EBBP", "BASP", "BASPN"]

add_finta_feature(data_min, data_finta, extra_features, both_columns_features)


# %%
plot_correlation(data_min,
                     title="Correlations Between Extended Features in Minutes",
                     figsize=(46, 23))

# %%
random_start_point = False
clip_param = 0.75
lr = 0.5
gamma = 0.95
step_size = 1.0

# %%
start_index = max_index + 1
data_min = data_min.loc[start_index:].reset_index(drop=True)

start_hour, start_minute = divmod(start_index, 60)

if 'Timestamp' in data_min.columns:
    data_min = data_min.drop('Timestamp', axis=1)

ordered_columns = ['Closing_price', 'Volume_of_transactions', 'Opening_price', 'Highest_price', 'Lowest_price', 'TRIX', 'VWAP', 'MACD',
                   'Sentiment_score', 'EV_MACD', 'MOM', 'RSI', 'IFT_RSI', 'TR', 'ATR', 'BBWIDTH', 'DMI_1', 'DMI_2', 'ADX', 'STOCHRSI', 'MI',
                   'CHAIKIN', 'VZO', 'PZO', 'EFI', 'EBBP_1', 'EBBP_2', 'BASP_1', 'BASP_2', 'BASPN_1', 'BASPN_2', 'WTO', 'SQZMI', 'VFI', 'STC']

data_min = data_min[ordered_columns]

if plot_data_process:
    data_min.info()


# %%
def normalize_data(train, val, test, scaler=pp.StandardScaler()):
    scaler_fitted = scaler.fit(train)

    train = torch.from_numpy(scaler_fitted.transform(train))
    val = torch.from_numpy(scaler_fitted.transform(val)) if val is not None else None
    test = torch.from_numpy(scaler_fitted.transform(test))

    return train, val, test, scaler_fitted


# %%
def train_validation_test_split(data, val_percentage, test_percentage):
    total = len(data)
    split_points = [int(total * p) for p in (val_percentage, test_percentage)]
    
    train_end = total - sum(split_points)
    val_end = train_end + split_points[0]

    train = data.iloc[:train_end]
    val = data.iloc[train_end:val_end] if split_points[0] > 0 else None
    test = data.iloc[val_end:]

    return train, val, test


# %%
def betchify(data, batch_size):
    total_len = (data.size(0) // batch_size) * batch_size
    trimmed = data.narrow(0, 0, total_len)
    reshaped = trimmed.reshape(batch_size, -1, data.size(1))
    reordered = reshaped.permute(1, 0, 2).contiguous()
    return reordered.to(device)


# %%
plot_data_process = True


num_features = 34 
scaler_name = 'minmax'
train_batch_size = 32
eval_batch_size = 32
epochs = 50
bptt_src = 10
bptt_tgt = 2
overlap = 1

# %%
def get_batch(data, i, bptt_src, bptt_tgt, overlap):
    max_src = len(data) - i - 1
    src_seq_len = bptt_src if bptt_src < max_src else max_src

    tgt_start = i + src_seq_len - overlap
    max_tgt = len(data) - tgt_start
    target_seq_len = bptt_tgt if bptt_tgt < max_tgt else max_tgt

    source = data[i : i + src_seq_len]
    target = data[tgt_start : tgt_start + target_seq_len]
    return source, target


# %%
class SineActivation(nn.Module):
    def __init__(self, in_features, periodic_features, out_features, dropout):
        super(SineActivation, self).__init__()

        extra_dim = out_features - in_features - periodic_features

        self.w0 = nn.Parameter(torch.empty(in_features, extra_dim).normal_())
        self.b0 = nn.Parameter(torch.empty(1, extra_dim).uniform_())

        self.w = nn.Parameter(torch.empty(in_features, periodic_features).normal_())
        self.b = nn.Parameter(torch.empty(1, periodic_features).uniform_())

        self.activation = torch.sin
        self.dropout = nn.Dropout(dropout)

    def Time2Vector(self, data):
        x = data.permute(0, 2, 1)

        linear_out = torch.bmm(x, self.w0.unsqueeze(0).expand(x.size(0), -1, -1)).permute(0, 2, 1) + self.b0
        periodic_out = self.activation(torch.bmm(x, self.w.unsqueeze(0).expand(x.size(0), -1, -1)).permute(0, 2, 1) + self.b)

        return torch.cat([linear_out, periodic_out, data], dim=2)

    def forward(self, data):
        transformed = self.Time2Vector(data)
        return self.dropout(transformed)


# %%
class BTC_Transformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 in_features: int,
                 periodic_features: int,
                 out_features: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super(BTC_Transformer, self).__init__()

        self.sine_activation = SineActivation(in_features, periodic_features, out_features, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=out_features,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation)

        decoder_layer = nn.TransformerDecoderLayer(d_model=out_features,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(out_features, in_features)

    def encode(self, src: Tensor, src_mask: Tensor):
        embedded_src = self.sine_activation(src)
        return self.encoder(embedded_src, mask=src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        embedded_tgt = self.sine_activation(tgt)
        return self.decoder(embedded_tgt, memory, tgt_mask)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor = None,
                tgt_mask: Tensor = None,
                mem_mask: Tensor = None,
                src_padding_mask: Tensor = None,
                tgt_padding_mask: Tensor = None,
                memory_key_padding_mask: Tensor = None):

        src_proj = self.sine_activation(src)
        tgt_proj = self.sine_activation(trg)

        output = nn.Transformer(d_model=src_proj.size(-1),
                                nhead=src_proj.size(-1) // nhead,
                                num_encoder_layers=0,
                                num_decoder_layers=0)(src_proj, tgt_proj)

        output = self.transformer(src_proj, tgt_proj,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  memory_mask=mem_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)

        return self.generator(output)


# %%
def evaluate(model, data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature):
    model.eval()
    total_loss = 0.0

    base_src_mask = torch.zeros((bptt_src, bptt_src), dtype=torch.bool, device=device)
    base_tgt_mask = model.transformer.generate_square_subsequent_mask(bptt_tgt).to(device)

    with torch.no_grad():
        positions = range(0, data.size(0) - 1, bptt_src)
        for i in positions:
            source, targets = get_batch(data, i, bptt_src, bptt_tgt, overlap)

            if source.size(0) != bptt_src:
                src_mask = base_src_mask[:source.size(0), :source.size(0)]
            else:
                src_mask = base_src_mask

            if targets.size(0) != bptt_tgt:
                tgt_mask = base_tgt_mask[:targets.size(0), :targets.size(0)]
            else:
                tgt_mask = base_tgt_mask

            prediction = model(source, targets, src_mask, tgt_mask)
            step_loss = criterion(prediction[:-1, :, predicted_feature], targets[1:, :, predicted_feature])
            total_loss += step_loss.item() * source.size(0)

    mean_loss = total_loss / (data.size(0) - 1)
    return mean_loss


# %%
val_percentage = 0.1
test_percentage = 0.1
train_df, val_df, test_df = train_validation_test_split(data_min, val_percentage, test_percentage)
print(np.shape(train_df))
if val_df is not None:
    print(np.shape(val_df))
print(np.shape(test_df))

# %%
import matplotlib.pyplot as plt
import numpy as np

if plot_data_process:
    t0 = np.size(train_df, 0)
    tv = np.size(val_df, 0) if val_df is not None else 0
    tt = np.size(test_df, 0)

    train_time = np.arange(t0)
    val_time = np.arange(t0, t0 + tv) if val_df is not None else None
    test_time = np.arange(t0 + tv, t0 + tv + tt)

    fig, ax = plt.subplots(figsize=(18, 9))
    fig.suptitle("Time-Series Data for Price Prediction", fontsize=22, fontweight='bold')

    ax.set_title(" Price Across Training, Validation, and Test Sets", fontsize=16)
    ax.set_xlabel(f"Time (Minutes)", fontsize=14)
    ax.set_ylabel(" Price", fontsize=14)

    ax.plot(train_time, train_df['Closing_price'], label='Training Data', color='tab:blue', linewidth=1.8)
    
    if val_df is not None:
        ax.plot(val_time, val_df['Closing_price'], label='Validation Data', color='tab:orange', linewidth=1.8)
    
    ax.plot(test_time, test_df['Closing_price'], label='Test Data', color='tab:red', linewidth=1.8)

    ax.grid(visible=True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize=13)

    stats = [
        f"Train Samples: {len(train_df)}",
        f"Validation Samples: {len(val_df) if val_df is not None else 0}",
        f"Test Samples: {len(test_df)}"
    ]
    box_config = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.5, 0.95, "\n".join(stats), transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=box_config)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# %%
predicted_feature = train_df.columns.get_loc('Closing_price')

scaler = pp.StandardScaler() if scaler_name == 'standard' else pp.MinMaxScaler()

in_features = num_features
criterion = nn.MSELoss()

model = BTC_Transformer(
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    in_features=in_features,
    periodic_features=periodic_features,
    out_features=out_features,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    activation=activation
).to(device)

optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train = train_df.iloc[:, :num_features]
val = val_df.iloc[:, :num_features] if val_df is not None else None
test = test_df.iloc[:, :num_features]

train, val, test, scaler = normalize_data(train, val, test, scaler)

train_data = betchify(train, train_batch_size).float()
val_data = betchify(val, eval_batch_size).float() if val is not None else None
test_data = betchify(test, eval_batch_size).float()


# %%
# train the model
best_val_loss = float('inf')
best_model = None
train_loss = []
valid_loss = []

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0.0
    accumulated_loss = 0.0
    start_time = time.time()

    start_point = np.random.randint(bptt_src) if random_start_point else 0
    total_batches = (len(train_data) - start_point) // bptt_src
    log_interval = max(10, (total_batches // 30) * 10)

    src_mask_template = torch.zeros((bptt_src, bptt_src), dtype=torch.bool, device=device)
    tgt_mask_template = model.transformer.generate_square_subsequent_mask(bptt_tgt).to(device)

    for batch_idx, idx in enumerate(range(start_point, train_data.size(0) - 1, bptt_src)):
        source, target = get_batch(train_data, idx, bptt_src, bptt_tgt, overlap)

        src_len, tgt_len = source.size(0), target.size(0)
        current_src_mask = src_mask_template[:src_len, :src_len]
        current_tgt_mask = tgt_mask_template[:tgt_len, :tgt_len]

        prediction = model(source, target, current_src_mask, current_tgt_mask)
        loss = criterion(prediction[:-1, :, predicted_feature], target[1:, :, predicted_feature])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_param)
        optimizer.step()

        total_loss += loss.item()
        accumulated_loss += src_len * loss.item()

        if (batch_idx % log_interval == 0) and batch_idx > 0:
            lr = scheduler.get_last_lr()[0]
            time_per_batch = (time.time() - start_time) * 1000 / log_interval
            avg_loss = total_loss / log_interval
            print(f'| epoch {epoch:3d} | {batch_idx:5d}/{total_batches:5d} batches | '
                  f'lr {lr:.6f} | ms/batch {time_per_batch:5.2f} | '
                  f'loss {avg_loss:.6f}')
            total_loss = 0.0
            start_time = time.time()

    train_loss.append(accumulated_loss / (len(train_data) - 1))

    if val is not None:
        val_loss = evaluate(model, val_data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature)
        epoch_duration = time.time() - epoch_start_time

        print('-' * 77)
        print(f'| end of epoch {epoch:3d} | time: {epoch_duration:5.2f}s | '
              f'valid loss {val_loss:.6f}')
        print('-' * 77)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        valid_loss.append(val_loss)

    scheduler.step()

if val is None:
    best_model = copy.deepcopy(model)


# %%
def estimate_BTC(best_model, test, num_features, bptt_src, bptt_tgt, overlap, predicted_feature, scaler, use_real=True, early_stop=1):
    inference_bptt_src = bptt_src + int(overlap == 0)
    inference_batch_size = 1
    step_size = min(bptt_tgt - overlap, bptt_tgt - 1)

    test_data = betchify(test[:, :num_features], inference_batch_size).float()
    total_steps = (test_data.size(0) - bptt_src) // step_size

    data_window = test_data[:inference_bptt_src]

    collected = []

    for i in range(total_steps):
        pred = greedy_decode(best_model, data_window, bptt_src, step_size, overlap)

        if use_real:
            idx_start = i * step_size
            idx_end = idx_start + inference_bptt_src
            data_window = test_data[idx_start:idx_end]
        else:
            data_window = torch.cat([data_window, pred], dim=0)[step_size:]

        collected.append(pred)

        if i > total_steps * early_stop:
            break

    predictions = torch.cat(collected, dim=0)

    true_values = scaler.inverse_transform(
        test_data.permute(1, 0, 2).reshape(-1, num_features).cpu()
    )[:, predicted_feature]

    predicted_values = scaler.inverse_transform(
        predictions.permute(1, 0, 2).reshape(-1, num_features).cpu()
    )[:, predicted_feature]

    return true_values, predicted_values, inference_bptt_src


# %%
early_stop = 1

feature_real, feature_prediction, pred_start = estimate_BTC(
    best_model, test, num_features, bptt_src, bptt_tgt,
    overlap, predicted_feature, scaler,
    use_real=True, early_stop=early_stop
)

real_time = np.arange(len(feature_real))
prediction_time = np.arange(pred_start, pred_start + len(feature_prediction))


# %%
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_title("Closing price through time")
ax.set_xlabel("Time [Minutes]")
ax.set_ylabel("Closing Price [USD]")
ax.grid(visible=True)

ax.plot(real_time, feature_real, label='Real Values')
ax.plot(prediction_time, feature_prediction, label='Predicted Values')

x_range = len(prediction_time)
y_min = min(np.min(feature_prediction), np.min(feature_real[:x_range]))
y_max = max(np.max(feature_prediction), np.max(feature_real[:x_range]))

ax.set_xlim(0, x_range)
ax.set_ylim(y_min, y_max)

ax.legend()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

shift = 3
feature_prediction_aligned = feature_prediction.copy()
feature_prediction_aligned[:-shift] = feature_prediction[shift:]
feature_prediction_aligned[-shift:] = np.nan

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_title("Closing price through time - Zoom In")
ax.set_xlabel("Time [Minutes]")
ax.set_ylabel("Closing Price [USD]")
ax.grid(True)

ax.plot(real_time, feature_real, label='Real Values')
ax.plot(prediction_time, feature_prediction_aligned, label='Predicted Values (aligned)')

min_val = min(np.nanmin(feature_prediction_aligned[low:high]), np.nanmin(feature_real[low:high]))
max_val = max(np.nanmax(feature_prediction_aligned[low:high]), np.nanmax(feature_real[low:high]))

ax.set_xlim(left=low, right=high)
ax.set_ylim(bottom=min_val, top=max_val)

ax.legend()
plt.show()


# %%
import pickle

# To save
with open('feature_prediction.pkl', 'wb') as f:
    pickle.dump(feature_prediction, f)

# To load (in another session)
with open('feature_prediction.pkl', 'rb') as f:
    feature_prediction = pickle.load(f)


# %%
import pickle

# To save
with open('feature_real.pkl', 'wb') as f:
    pickle.dump(feature_real, f)

# To load (in another session)
with open('feature_real.pkl', 'rb') as f:
    feature_real = pickle.load(f)

# %%
import pickle

# To save
with open('real_time.pkl', 'wb') as f:
    pickle.dump(real_time, f)

# To load (in another session)
with open('real_time.pkl', 'rb') as f:
    real_time = pickle.load(f)

# %%
capital = 1000  # Initial USD
btc = 0         # Initial BTC

for i in range(1, len(feature_prediction)):
    if feature_prediction[i] > feature_real[i-1]:  # BUY signal
        if capital > 0:
            btc = capital / feature_real[i]       # buy BTC
            capital = 0
    else:  # SELL signal
        if btc > 0:
            capital = btc * feature_real[i]       # sell BTC
            btc = 0

# Final value (assuming all BTC sold at last price)
final_value = capital + btc * feature_real[-1]
roi = (final_value - 1000) / 1000 * 100
print(f"Final portfolio value: ${final_value:.2f}")
print(f"Return on investment: {roi:.2f}%")


# %% [markdown]
# Zoomed In

# %%

# Define the zoom-in range on the test data
low = 42550
high = 42650

# Track state and generate signals
holding_btc = False  # Initially not holding BTC
buy_points = []
sell_points = []

for i in range(low, high):
    if i < 1:
        continue
    if not holding_btc and feature_prediction[i] > feature_real[i - 1]:
        buy_points.append(i)
        holding_btc = True
    elif holding_btc and feature_prediction[i] <= feature_real[i - 1]:
        sell_points.append(i)
        holding_btc = False

# Plot real BTC price with signals
plt.figure(figsize=(14, 6))
plt.plot(real_time[low:high], feature_real[low:high], label='Real BTC Price', linewidth=2)

plt.scatter(buy_points, feature_real[buy_points], marker='^', color='green', label='Buy Signal', s=100)
plt.scatter(sell_points, feature_real[sell_points], marker='v', color='red', label='Sell Signal', s=100)

plt.xlabel('Time (minutes)')
plt.ylabel('BTC Closing Price [USD]')
plt.title('Buy/Sell Signals with Alternating Trades (Zoomed In)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# Whole test data

# %%

in_position = False
buy_points = []
sell_points = []

for i in range(1, len(feature_prediction)):
    if not in_position and feature_prediction[i] > feature_real[i - 1]:
        buy_points.append(i)
        in_position = True
    elif in_position and feature_prediction[i] < feature_real[i - 1]:
        sell_points.append(i)
        in_position = False

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(np.arange(len(feature_real)), feature_real, label='Real BTC Price', linewidth=2)

plt.scatter(buy_points, feature_real[buy_points], marker='^', color='green', label='Buy Signal', s=100)
plt.scatter(sell_points, feature_real[sell_points], marker='v', color='red', label='Sell Signal', s=100)

plt.xlabel('Time (minutes)')
plt.ylabel('BTC Closing Price [USD]')
plt.title('BTC Price with Alternating Buy/Sell Signals on Test Set')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%

capital = 1000
btc = 0
in_position = False
buy_points = []
sell_points = []
portfolio_values = []

# Adjusted lists
adjusted_prices = []

for i in range(1, len(feature_prediction)):
    price = feature_real[i]
    adjusted_prices.append(price)  # aligned with portfolio tracking
    
    if not in_position and feature_prediction[i] > feature_real[i - 1]:
        btc = capital / price
        capital = 0
        in_position = True
        buy_points.append(i - 1)  # correct index
    elif in_position and feature_prediction[i] < feature_real[i - 1]:
        capital = btc * price
        btc = 0
        in_position = False
        sell_points.append(i - 1)  # correct index
    
    portfolio_value = capital + btc * price
    portfolio_values.append(portfolio_value)

# Fix time axis
plot_time = np.arange(1, len(portfolio_values) + 1)

# === Combined Graph ===
fig, ax1 = plt.subplots(figsize=(14, 6))

# BTC Price
ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('BTC Price [USD]', color='tab:blue')
ax1.plot(plot_time, adjusted_prices, label='Real BTC Price', color='tab:blue', linewidth=2)
ax1.scatter(np.array(buy_points) + 1, np.array(feature_real)[np.array(buy_points) + 1], marker='^', color='green', label='Buy Signal', s=100)
ax1.scatter(np.array(sell_points) + 1, np.array(feature_real)[np.array(sell_points) + 1], marker='v', color='red', label='Sell Signal', s=100)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

# Portfolio Value
ax2 = ax1.twinx()
ax2.set_ylabel('Portfolio Value [USD]', color='tab:purple')
ax2.plot(plot_time, portfolio_values, label='Portfolio Value', color='tab:purple', linewidth=2, linestyle='--')
ax2.tick_params(axis='y', labelcolor='tab:purple')

# Legends
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)
plt.title('BTC Price with Buy/Sell Signals and Portfolio Value')
plt.tight_layout()
plt.show()


# %%
# === STRICT ALTERNATING BUY/SELL STRATEGY & PORTFOLIO TRACKING ON TEST SET ===

capital = 1000  # initial USD
btc = 0         # initial BTC
in_position = False
buy_points = []
sell_points = []
portfolio_values = []

# Track only the valid time indices
price_indices = []

i = 1
while i < len(feature_prediction):
    price = feature_real[i]

    if not in_position and feature_prediction[i] > feature_real[i - 1]:
        # Buy
        btc = capital / price
        capital = 0
        in_position = True
        buy_points.append(i)

        # Hold until a sell condition
        while i < len(feature_prediction):
            price = feature_real[i]
            portfolio_values.append(capital + btc * price)
            price_indices.append(i)

            if feature_prediction[i] < feature_real[i - 1]:
                # Sell
                capital = btc * price
                btc = 0
                in_position = False
                sell_points.append(i)
                i += 1
                break
            i += 1
    else:
        portfolio_values.append(capital + btc * price)
        price_indices.append(i)
        i += 1

# If still holding, track remaining value
while i < len(feature_real):
    price = feature_real[i]
    portfolio_values.append(capital + btc * price)
    price_indices.append(i)
    i += 1

# === PREP FOR PLOTTING ===
plot_time = np.arange(len(portfolio_values))
matched_prices = feature_real[price_indices]

# Ensure signals only appear in range
buy_points_filtered = [i for i in buy_points if i in price_indices]
sell_points_filtered = [i for i in sell_points if i in price_indices]

# Re-index signals to plot_time coordinates
index_map = {idx: i for i, idx in enumerate(price_indices)}
buy_plot = [index_map[i] for i in buy_points_filtered]
sell_plot = [index_map[i] for i in sell_points_filtered]

# === PLOTTING BTC PRICE & PORTFOLIO VALUE ===
fig, ax1 = plt.subplots(figsize=(14, 6))

# BTC price
ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('BTC Price [USD]', color='tab:blue')
ax1.plot(plot_time, matched_prices, color='tab:blue', label='Real BTC Price', linewidth=2)
ax1.scatter(buy_plot, matched_prices[buy_plot], marker='^', color='green', label='Buy Signal', s=100)
ax1.scatter(sell_plot, matched_prices[sell_plot], marker='v', color='red', label='Sell Signal', s=100)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

# Portfolio value (right axis)
ax2 = ax1.twinx()
ax2.set_ylabel('Portfolio Value [USD]', color='tab:purple')
ax2.plot(plot_time, portfolio_values, color='tab:purple', label='Portfolio Value', linewidth=2, linestyle='--')
ax2.tick_params(axis='y', labelcolor='tab:purple')

# Title & legend
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)
plt.title('BTC Price, Buy/Sell Signals, and Portfolio Value on Test Set')
plt.tight_layout()
plt.show()



# %%
import plotly.graph_objects as go
import numpy as np

# Prepare data
plot_time = np.arange(len(portfolio_values))
matched_prices = feature_real[price_indices]

# Map buy/sell to local indices for plotting
index_map = {idx: i for i, idx in enumerate(price_indices)}
buy_plot = [index_map[i] for i in buy_points if i in index_map]
sell_plot = [index_map[i] for i in sell_points if i in index_map]

# === Create interactive Plotly graph ===
fig = go.Figure()

# BTC Price
fig.add_trace(go.Scatter(x=plot_time, y=matched_prices,
                         mode='lines', name='Real BTC Price',
                         line=dict(color='blue')))

# Buy signals
fig.add_trace(go.Scatter(x=buy_plot, y=matched_prices[buy_plot],
                         mode='markers', name='Buy',
                         marker=dict(symbol='triangle-up', size=10, color='green')))

# Sell signals
fig.add_trace(go.Scatter(x=sell_plot, y=matched_prices[sell_plot],
                         mode='markers', name='Sell',
                         marker=dict(symbol='triangle-down', size=10, color='red')))

# Portfolio value (right y-axis)
fig.add_trace(go.Scatter(x=plot_time, y=portfolio_values,
                         mode='lines', name='Portfolio Value',
                         yaxis='y2',
                         line=dict(color='purple', dash='dash')))

# Layout with dual y-axes
fig.update_layout(
    title='Zoomable BTC Price and Portfolio Value (Test Set)',
    xaxis=dict(title='Time (minutes)'),
    yaxis=dict(
        title=dict(text='BTC Price [USD]', font=dict(color='blue')),
        tickfont=dict(color='blue')
    ),
    yaxis2=dict(
        title=dict(text='Portfolio Value [USD]', font=dict(color='purple')),
        tickfont=dict(color='purple'),
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0.01, y=0.99),
    hovermode='x unified',
    template='plotly_white',
    height=600
)


fig.write_html("btc_strategy_plot1.html", auto_open=True)



# %%
import plotly.graph_objects as go
import numpy as np

# === STRICT ALTERNATING BUY/SELL STRATEGY ON TEST DATA ===

capital = 1000
btc = 0
in_position = False
buy_points = []
sell_points = []
portfolio_values = []
price_indices = []

for i in range(1, len(feature_prediction)):
    price = feature_real[i]
    pred = feature_prediction[i]

    if not in_position and pred > feature_real[i - 1]:
        # BUY
        btc = capital / price
        capital = 0
        in_position = True
        buy_points.append(i)

    elif in_position and pred < feature_real[i - 1]:
        # SELL
        capital = btc * price
        btc = 0
        in_position = False
        sell_points.append(i)

    # Always record current portfolio value
    portfolio_values.append(capital + btc * price)
    price_indices.append(i)

# === PREP FOR PLOTTING ===
plot_time = np.arange(len(portfolio_values))
matched_prices = feature_real[price_indices]

# Map real indices to plot indices
index_map = {real_i: plot_i for plot_i, real_i in enumerate(price_indices)}
buy_plot = [index_map[i] for i in buy_points if i in index_map]
sell_plot = [index_map[i] for i in sell_points if i in index_map]

# === PLOTLY CHART (ZOOMABLE, INTERACTIVE) ===
fig = go.Figure()

# BTC price line
fig.add_trace(go.Scatter(
    x=plot_time, y=matched_prices,
    mode='lines', name='Real BTC Price',
    line=dict(color='blue')
))

# Buy markers
fig.add_trace(go.Scatter(
    x=buy_plot, y=matched_prices[buy_plot],
    mode='markers', name='Buy',
    marker=dict(symbol='triangle-up', size=6, color='green')
))

# Sell markers
fig.add_trace(go.Scatter(
    x=sell_plot, y=matched_prices[sell_plot],
    mode='markers', name='Sell',
    marker=dict(symbol='triangle-down', size=6, color='red')
))

# Portfolio value (second y-axis)
fig.add_trace(go.Scatter(
    x=plot_time, y=portfolio_values,
    mode='lines', name='Portfolio Value',
    yaxis='y2',
    line=dict(color='purple', dash='dash')
))

# Layout with zoom tools and dual axes
fig.update_layout(
    title='BTC Price, Buy/Sell Signals, and Portfolio Value (Test Set)',
    xaxis=dict(
        title='Time (index)',
        rangeslider=dict(visible=True),
        rangeselector=dict(
            buttons=[
                dict(count=1000, label='1000 pts', step='second', stepmode='backward'),
                dict(count=5000, label='5000 pts', step='second', stepmode='backward'),
                dict(step='all', label='All')
            ]
        )
    ),
    yaxis=dict(
        title=dict(text='BTC Price [USD]', font=dict(color='blue')),
        tickfont=dict(color='blue')
    ),
    yaxis2=dict(
        title=dict(text='Portfolio Value [USD]', font=dict(color='purple')),
        tickfont=dict(color='purple'),
        overlaying='y',
        side='right'
    ),
    hovermode='x unified',
    template='plotly_white',
    height=600
)

# Open interactive chart in browser
fig.write_html("btc_strategy_plot2.html", auto_open=True)


# %%
# Hyperparameters
capital = 1000
btc = 0
in_position = False
buy_price = 0
profit_target = 0.015   # +1.5%
stop_loss = -0.0075     # -0.75%

buy_points = []
sell_points = []
portfolio_values = []
adjusted_prices = []

for i in range(1, len(feature_prediction)):
    price = feature_real[i]
    pred_price = feature_prediction[i]
    adjusted_prices.append(price)

    # Buy condition: prediction > previous real price
    if not in_position and pred_price > feature_real[i - 1]:
        btc = capital / price
        buy_price = price
        capital = 0
        in_position = True
        buy_points.append(i)
    
    # Sell conditions
    elif in_position:
        gain_pct = (price - buy_price) / buy_price
        if gain_pct >= profit_target or gain_pct <= stop_loss:
            capital = btc * price
            btc = 0
            in_position = False
            sell_points.append(i)

    portfolio_value = capital + btc * price
    portfolio_values.append(portfolio_value)

# Plotting (same as before)
plot_time = np.arange(1, len(portfolio_values) + 1)
fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('BTC Price [USD]', color='tab:blue')
ax1.plot(plot_time, adjusted_prices, label='BTC Price', color='tab:blue', linewidth=2)
ax1.scatter(np.array(buy_points), np.array(feature_real)[np.array(buy_points)], marker='^', color='green', label='Buy', s=100)
ax1.scatter(np.array(sell_points), np.array(feature_real)[np.array(sell_points)], marker='v', color='red', label='Sell', s=100)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel('Portfolio Value [USD]', color='tab:purple')
ax2.plot(plot_time, portfolio_values, label='Portfolio Value', color='tab:purple', linewidth=2, linestyle='--')
ax2.tick_params(axis='y', labelcolor='tab:purple')

fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)
plt.title('BTC Price with Profit-Target/Stop-Loss Strategy')
plt.tight_layout()
plt.show()


# %%




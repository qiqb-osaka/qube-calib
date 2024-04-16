# qube-calib を使ってみる

## 環境構築

### 仮装環境の作成

任意の作業ディレクトリで以下の手順を実行すると，新しい仮装環境が利用可能な状態になる．

```shell
python3.9 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### qube-calib のインストール

以下のコマンドで qube-calib をインストールする．

```shell
pip install -r /usr/local/qube/pipfiles/requirements-qubecalib-3.0.0-beta.txt
```

### quelware のインストール

qube001 上では，以下のコマンドで quelware 0.8.8 をインストールする．

```
cd /usr/local/qube/package/quelware/quel_ic_config
pip install -r requirements_simplemulti_standard.txt
cd -
```

それ以外の場合，以下の指示に従って quelware をインストールする．qube001 上では，コンパイル済みバッケージの取得と展開以降を /usr/local/qube/package/quelware/ 以下に実施済みである．

`https://github.com/quel-inc/sugita_experimental/blob/main/quel_ic_config/GETTING_STARTED.md`

ここで，コンパイル済みバッケージの取得と展開以降を実施し，SIMPLEMULTI_STANDARDの場合をインストールすれば良い．

### VScode で作業ディレクトリを開く

例えばリモートエクスプローラーから qube001 へ接続し，作業ディレクトリを開く

# 信号の送受信をしてみる

Qube Riken 機には測定用受信機 2 系統とモニタ用受信機 2 系統が備わっている．

## 例題スクリプト

### 設定スクリプト

- `0a_experimental_setup.ipynb` 装置配線などを設定して保存する．保存した内容を実験スクリプトでロードして使う．
- `0b_config_box.ipynb` 装置の LSI を設定して保存する．装置の状態を復元したい場合に load and apply する．

### 実験スクリプト

- `1_myfirst_pulse_sequence.ipynb` 単体筐体で網羅的に信号を検出する



# API

- 装置設定
    - create_box(): Box オブジェクトを生成する
    -

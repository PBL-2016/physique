# 体格分析システム

### 要件

+ 身体画像から体格を認識・分析

入力:画像ファイル名(コマンドライン引数)
出力:上下比, 縦横比, 肌色, 髪色, 肩タイプ, 顔タイプ

上下比   -> 上半身の高さ(px) / 下半身の高さ(px)
縦横比   -> 身体高さ(px) / 胴幅(px)
肌色     -> 0xRRGGBB
髪色     -> 0xRRGGBB
肩タイプ -> N(撫で肩) or I(怒り肩)
顔タイプ -> C(●型) or S(■型) or T(▲型) or U(▼型)


## Mac Requirements

```bash
$ brew tap homebrew/science
$ brew install openblas
$ brew install hdf5
$ pip install numpy
$ brew install opencv3 --with-python3
$ brew link opencv3 -force
$ pip install chainer
$ pip install h5py
```

## Usage

```bash
$ python3 predict.py foo.jpg
```
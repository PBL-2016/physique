追加データの加工手順(全体画像+上半身切り抜き画像)


解析
```
$ python extract_ignorable.py 全体画像ディレクトリ > data1.csv
```

解析結果の幅比を補正
Process.ipynb data1.csv -> data2.csv

タグ付けデータと解析結果を結合
Join.ipynb data.csv data2.csv -> joined.csv

モデル訓練
Train.ipynb -> pbllogi.model pbllogi.state
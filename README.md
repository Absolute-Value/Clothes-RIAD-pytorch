# ファイルとフォルダの説明


## losses/
RIADにて使用されるGMSとSSIMのLossのプログラムです。

## modules/
* *funcs.py*
    * 早期終了等のプログラムです。
* *initializer.py*
    * 初期化のプログラムです。
* *loader.py*
    * データ読み込むプログラムです。
* *logger.py*
    * ログ出力のプログラムです。
* *unet.py*
    * UNETのネットワークのプログラムです。

## data/
* *other.py*
    * データごとの読み込みのプログラムです
* *other256/wrench/train/good*
    * 学習データを入れてください
* *other256/wrench/test/good*
    * テストデータの正常のものを入れてください
* *other256/wrench/test/bad*
    * テストデータの異常のものを入れてください

## result/
結果を出力するフォルダで、実行すると自動生成されます  
*other/* 内に結果が入っています

以下、出力内部詳細
* *train.csv*
    * 学習時のパラメータとLossが保存されています
* *train.log*
    * 学習時のログ出力です
* *test.csv*
    * テスト時のパラメータとAUCが保存されています
* *test.log*
    * テスト時のログ出力です
* *model.pt*
    * モデルが保存されています
* *loss.png*
    * Lossのグラフです
* *ad/*
    * 異常検知結果の画像が入っています(入出力、AUCなど)
* *pic/*
    * 学習時のTrain, Validation入出力画像が入っています

## train.py
学習のプログラムです。

## test.py
テスト（異常検知）のプログラムです

## run.sh
学習を実行するスクリプトです。

## makeTest.sh
テストを実行するスクリプトを出力するプログラムです

# 使用方法
1. *data/other256/wrench/* 内に画像データを配置してください
    * 詳細はフォルダの説明 *data/* をご覧ください
2. 学習は *python train.py --dataset other --data_type wrench* で開始します
    * linuxであれば*bash run.sh*でも実行できます
3. 学習は *python test.py --save_dir (結果フォルダ名：result/other/wrench256_seed999_20210720_022746等)* で開始します
    * linuxであれば*python makeTestSh.py*を実行して、*bash test.sh* でも実行できます
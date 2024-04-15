# Wishlist

- CaptureParams の DSP を設定する Command を追加する
    - TARGET 毎に指定できるようにする
        - CAPTURE 用途の TARGET 以外に定義された場合は exec() 時にエラーを出す
            - neopulse のレベルでエラーにする方が良いかも？
    - この設定は exec() で指定した
- データベースの定義と保存

# Todo

- README.md の更新
```
$ cd <your-workspace>
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

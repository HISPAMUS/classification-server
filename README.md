# classification-server
REST API offering services for these classifiers:
- symbol classification: commit [ffef905](https://github.com/HISPAMUS/symbol-classification/commit/ffef905002a2964e6a56d7c5d6b81c43e37a3a42)
- end-to-end recognition: commit [e18d694](https://github.com/HISPAMUS/end-to-end-recognition/commit/e18d694b3081ba5e95fdca3009b04e1a72e97795)

## Requirements

- Python 3.6
- $ pip install -r requirements.txt

## Usage

Can be tested from *grfia* with:
```bash
curl -F left=962 -F top=148 -F right=1050 -F bottom=292 -F predictions=5 deep:8888/image/symbol_test/symbol
curl deep:8888/image/e2e_test/e2e
````

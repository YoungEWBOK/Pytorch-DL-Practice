## 项目结构
- `train.py`：用于训练模型
- `test.py`：用于测试模型
- `main.py`：整合训练和测试的执行
- `config.py`：配置文件，包含模型参数和训练设置
- `weights/`：保存模型权重的文件夹
- `runs/`：TensorBoard 日志目录
- `origin_data/`：初始数据集目录

## 执行步骤
安装依赖
```bash
pip install -r requirements.txt
```

数据集分割
```bash
python data_split.py
```

模型训练与验证
```bash
python main.py
```

train.py 和 test.py 可以单独运行，也可以通过 main.py 整合运行。

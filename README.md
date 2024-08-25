- `train.py`：用于训练模型
- `test.py`：用于测试模型
- `main.py`：整合训练和测试的执行
- `config.py`：配置文件，包含模型参数和训练设置
- `weights/`：保存模型权重的文件夹
- `runs/`：TensorBoard 日志目录


## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明
训练模型
要训练模型，可以直接运行 train.py 或通过 main.py 进行整合训练。以下是通过 main.py 的示例：

```bash
python main.py --train
```
测试模型
要测试训练好的模型，可以运行 test.py。确保 weights/ 文件夹中已有模型权重文件：

```bash
python test.py --weights weights/best_model.pth
```
查看 TensorBoard 日志
要查看训练过程中的 TensorBoard 日志，可以使用以下命令：

```bash
tensorboard --logdir=runs
```
然后在浏览器中访问 http://localhost:6006 查看可视化日志。

配置
模型参数和训练设置位于 config.py 文件中。你可以在此文件中修改训练批次大小、学习率等参数。

## 注意事项
训练过程中会保存模型的最佳权重到 weights/ 文件夹中，权重文件名包含日期信息。
确保在测试前，weights/ 文件夹中存在训练好的模型权重文件。
train.py 和 test.py 可以单独运行，也可以通过 main.py 整合运行。

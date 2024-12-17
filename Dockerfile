# 使用 Python 基礎映像
FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 複製當前目錄的所有文件到容器中的 /app 目錄
COPY . /app

# 安裝 virtualenv 工具
RUN pip install --no-cache-dir virtualenv

# 創建虛擬環境
RUN virtualenv venv

# 設定虛擬環境並安裝依賴
RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt

# 設定容器啟動時執行的命令
CMD ["bash", "-c", "source venv/bin/activate && python test.py"]

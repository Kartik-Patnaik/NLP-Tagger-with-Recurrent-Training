FROM python:3.6.8 
COPY requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY / /
EXPOSE 3131
ENTRYPOINT [ "python", "./new_stanford.py"]
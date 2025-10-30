#!/usr/bin/env bash
# 한 번만 초기화
if [ ! -f /opt/airflow/.initialized ]; then
  airflow db init
  airflow users create \
    --username senior --password 1224 \
    --firstname senior --lastname senior \
    --role Admin --email senior@example.com || true
  touch /opt/airflow/.initialized
fi

# 그 다음 항상 실행
exec airflow "$@"

#!/usr/bin/env bash
# 量化模型服务冒烟脚本
# 用法: PORT=8000 SERVED_NAME=quantized_model bash smoke_test.sh
set -euo pipefail

PORT="${PORT:?请设置 PORT}"
SERVED_NAME="${SERVED_NAME:?请设置 SERVED_NAME}"
BASE="http://127.0.0.1:${PORT}"

echo "[1/3] 连通性: GET /v1/models"
curl -sf "${BASE}/v1/models" | python3 -m json.tool | grep -q "${SERVED_NAME}" \
  && echo "  OK: ${SERVED_NAME} 已注册" \
  || { echo "  FAIL: 模型未注册"; exit 1; }

echo "[2/3] 生成抽检（中文/英文/代码各 1 条）"
for prompt in "用一句话介绍你自己。" "Explain quantization in one sentence." "写一个 Python 函数计算斐波那契数列第 n 项"; do
  resp=$(curl -sf "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":128,\"temperature\":0.6}" \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['message']['content'][:120].replace(chr(10),' '))")
  echo "  Q: ${prompt}"
  echo "  A: ${resp}"
done

echo "[3/3] 完成。请人工确认上面三条回答无乱码、无复读，再进入正式验收。"

#!/bin/bash
# 双击我：下载最新行情 → 用「最优拟合」的权重重算 → 打开网页
#
# 这一份是**最优拟合**，与 us2 那套生产版互不影响：
#   · 红点权重不同（杠杆30% / TOP2 20% / VIX绝对刻度20% / 上涨拥挤度20% / 前2%成交额占比10%，门槛 83）
#   · 没有黑框预警
#   · 没有温度走势图（其余图表面板与生产版一致）
# 这套权重是在 2017-10 至今那段历史上直接优选出来的，样本外没有证据支持——
# 页面「已知局限」第一条写明了这件事，看的时候务必先读它。

cd "$(dirname "$0")" || { echo "无法进入脚本所在目录"; exit 1; }

die() {
  echo
  echo "────────────────────────────────────────"
  echo "❌ $*"
  echo "────────────────────────────────────────"
  echo
  read -n 1 -s -r -p "按任意键关闭此窗口…"
  echo
  exit 1
}

echo "════════════════════════════════════════"
echo "  美股情绪温度计 · 最优拟合 · 更新并打开"
echo "  目录：$(pwd)"
echo "════════════════════════════════════════"
echo

# ---------- 0. 解除 macOS 隔离标记 ----------
xattr -dr com.apple.quarantine . 2>/dev/null || true

# ---------- 1. 找 python3 ----------
PY="$(command -v python3 2>/dev/null)"
[ -n "$PY" ] || die "找不到 python3。
   macOS 自带的命令行工具里就有，装一下即可：
       xcode-select --install
   装完重新双击本文件。"
echo "① Python：$PY（$("$PY" -V 2>&1)）"

# ---------- 2. 检查依赖 ----------
if "$PY" -c 'import pandas, numpy, scipy' 2>/dev/null; then
  echo "② 依赖：pandas / numpy / scipy 已就绪"
else
  echo "② 依赖缺失，正在安装 pandas / numpy / scipy…（首次可能要一两分钟）"
  if ! "$PY" -m pip install --user --quiet pandas numpy scipy 2>/dev/null \
     && ! "$PY" -m pip install --quiet pandas numpy scipy 2>/dev/null; then
    die "依赖安装失败。请手动在终端里跑：
       $PY -m pip install --user pandas numpy scipy
   装完再双击本文件。"
  fi
  "$PY" -c 'import pandas, numpy, scipy' 2>/dev/null \
    || die "依赖装完了但仍然导入失败，可能装到了另一个 Python 里。
   手动确认：$PY -c 'import pandas'"
  echo "   安装完成"
fi

# ---------- 3. 历史数据 ----------
if [ ! -d raw ] || [ -z "$(ls -A raw 2>/dev/null)" ]; then
  echo
  echo "③ ⚠️  本地还没有历史数据"
  echo "      需要下载 531 只标的 × 十年日线，约 40MB。"
  echo "      视网络情况 3–10 分钟，跨境链路可能更久。请耐心等待，别关窗口。"
else
  echo "③ 本地已有历史数据（raw/ 共 $(ls raw | wc -l | tr -d ' ') 个文件）"
fi

# ---------- 4. 跑流水线 ----------
echo
echo "④ 开始更新……"
echo "────────────────────────────────────────"
if [ "$SENTIMENT_FULL" = "1" ]; then
  MODE=""
  echo "   模式：全量（强制重下十年历史）"
else
  MODE="--fast"
  echo "   模式：增量（缺数据的标的会自动补全量）"
fi
echo

set -o pipefail
"$PY" alt_refresh.py $MODE 2>&1 | tee run.log
STATUS=$?
set +o pipefail

echo "────────────────────────────────────────"
[ $STATUS -eq 0 ] || die "更新失败（退出码 $STATUS）。
   上面的日志里有具体原因，也存在同目录的 run.log 里。
   常见情况：网络不通或数据源临时不可用——过一会儿再试通常就好了。"

# ---------- 5. 打开网页 ----------
[ -f dashboard_alt.html ] || die "流水线跑完了，但没有生成 dashboard_alt.html。
   请把 run.log 的内容发出来排查。"

echo
echo "⑤ 完成，正在打开网页…"
open dashboard_alt.html 2>/dev/null || die "网页生成好了但打不开。
   手动双击同目录下的 dashboard_alt.html 即可。"

echo
echo "✅ 全部完成。此窗口可以关掉了。"
echo
read -n 1 -s -r -t 20 -p "按任意键关闭（20 秒后自动关闭）…"
echo

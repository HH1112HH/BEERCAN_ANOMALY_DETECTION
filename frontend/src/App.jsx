import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowClockwise,
  ArrowSquareOut,
  Camera,
  CheckCircle,
  ClockCounterClockwise,
  Cpu,
  Gauge,
  Pause,
  Play,
  Pulse,
  SlidersHorizontal,
  WarningCircle,
  WifiHigh,
  WifiSlash,
  X,
} from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { apiRequest, apiUrl } from "./api.js";

const defaultThresholds = { patchcore: 0.0873, conf: 0.97 };
const enterTransition = { type: "spring", stiffness: 150, damping: 22 };

function Metric({ label, value, note, tone, icon: Icon, index = 0 }) {
  const reduceMotion = useReducedMotion();
  return (
    <motion.div
      className={`metric metric--${tone || "neutral"}`}
      initial={reduceMotion ? false : { opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ ...enterTransition, delay: index * 0.055 }}
      layout
    >
      <div className="metric__top">
        <span className="metric__label">{label}</span>
        {Icon && <Icon className="metric__icon" size={17} weight="regular" aria-hidden="true" />}
      </div>
      <strong className="metric__value">{value}</strong>
      <span className="metric__note">{note}</span>
    </motion.div>
  );
}

function Verdict({ value }) {
  const normalized = (value || "").toUpperCase();
  const isGood = normalized === "GOOD";
  const Icon = isGood ? CheckCircle : WarningCircle;
  return (
    <span className={`verdict verdict--${isGood ? "good" : "ng"}`}>
      <Icon size={15} weight="fill" aria-hidden="true" />
      {normalized || "—"}
    </span>
  );
}

function ResultRail({ history }) {
  const samples = history.slice(-14);
  if (!samples.length) {
    return <div className="sample-rail__empty">Chưa có mẫu kiểm tra</div>;
  }
  return (
    <div className="sample-rail" aria-label="Kết quả các lon gần nhất">
      {samples.map((item, index) => {
        const good = (item.verdict || "").toUpperCase() === "GOOD";
        return (
          <span
            className={`sample-rail__tick ${good ? "is-good" : "is-ng"}`}
            key={`${item.id}-${item.timestamp}-${index}`}
            title={`#${item.id} · ${item.verdict} · ${item.timestamp}`}
          />
        );
      })}
    </div>
  );
}

export default function App() {
  const reduceMotion = useReducedMotion();
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState("");
  const [cameraError, setCameraError] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [actionMessage, setActionMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [streamKey, setStreamKey] = useState(0);
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [thresholds, setThresholds] = useState(defaultThresholds);
  const [lastUpdated, setLastUpdated] = useState(null);
  const thresholdTimers = useRef({});
  const messageTimer = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const [nextStatus, nextHistory] = await Promise.all([
        apiRequest("/status"),
        apiRequest("/history"),
      ]);
      setStatus(nextStatus);
      setHistory(Array.isArray(nextHistory) ? nextHistory : []);
      setApiError("");
      setLastUpdated(new Date());
    } catch (error) {
      setApiError(error.message || "Không kết nối được API.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer;
    const poll = async () => {
      if (cancelled) return;
      await refresh();
      if (!cancelled) timer = window.setTimeout(poll, 1600);
    };
    poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      Object.values(thresholdTimers.current).forEach(window.clearTimeout);
      window.clearTimeout(messageTimer.current);
    };
  }, [refresh]);

  const announce = (message, kind = "success") => {
    setActionMessage(`${kind}:${message}`);
    window.clearTimeout(messageTimer.current);
    messageTimer.current = window.setTimeout(() => setActionMessage(""), 4200);
  };

  const submitAction = async (path, successText) => {
    setBusy(true);
    try {
      await apiRequest(path, { method: "POST" });
      announce(successText);
      await refresh();
    } catch (error) {
      announce(error.message || "Không thực hiện được thao tác.", "error");
    } finally {
      setBusy(false);
    }
  };

  const changeThreshold = (key, value) => {
    const numericValue = Number(value);
    setThresholds((current) => ({ ...current, [key]: numericValue }));
    window.clearTimeout(thresholdTimers.current[key]);
    thresholdTimers.current[key] = window.setTimeout(async () => {
      try {
        await apiRequest("/control/threshold", {
          method: "POST",
          body: JSON.stringify({ algo: key === "conf" ? "conf" : "patchcore", value: numericValue }),
        });
        announce("Đã cập nhật ngưỡng.");
      } catch (error) {
        announce(error.message || "Không cập nhật được ngưỡng.", "error");
      }
    }, 280);
  };

  const total = status?.total ?? 0;
  const hasSamples = total > 0;
  const yieldRate = hasSamples ? Number(status?.rate ?? 0) : null;
  const isConnected = Boolean(status?.conveyor_connected);
  const isProcessing = Boolean(status?.processing);
  const visibleHistory = useMemo(() => history.slice(-8).reverse(), [history]);
  const videoSrc = `${apiUrl("/video")}?stream=${streamKey}`;
  const updateLabel = lastUpdated
    ? lastUpdated.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : "Đang kết nối";

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Bỏ qua điều hướng</a>
      <header className="topbar">
        <a className="brand" href="#overview" aria-label="BeerCan Inspection, tổng quan">
          <span className="brand__mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <span className="brand__wordmark">BEERCAN<span> / VISION</span></span>
        </a>

        <nav className="topnav" aria-label="Điều hướng chính">
          <a className="topnav__link is-active" href="#overview" aria-current="page">Tổng quan</a>
          <a className="topnav__link" href="#inspections">Lịch sử</a>
          <a className="topnav__link" href="#controls">Điều khiển</a>
        </nav>

        <div className="topbar__right">
          <span className={`connection ${isConnected ? "is-online" : "is-offline"}`}>
            {isConnected ? <WifiHigh size={15} aria-hidden="true" /> : <WifiSlash size={15} aria-hidden="true" />}
            <span>{isConnected ? "ESP đã kết nối" : "ESP ngoại tuyến"}</span>
          </span>
          <span className="updated-at"><span className="updated-at__dot" />{updateLabel}</span>
        </div>
      </header>

      <main id="main-content" className="workspace">
        <AnimatePresence>
          {apiError && (
            <motion.div className="inline-alert" role="alert" initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <WarningCircle size={18} aria-hidden="true" />
              <span><strong>Mất kết nối API.</strong> {apiError}</span>
              <button className="icon-button" type="button" onClick={refresh} aria-label="Thử kết nối lại">
                <ArrowClockwise size={17} aria-hidden="true" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        <section id="overview" className="page-intro">
          <div>
            <p className="eyebrow"><span className="eyebrow__line" /> QUALITY CONTROL / LINE 01</p>
            <h1>Inspection overview</h1>
            <p className="page-intro__sub">Theo dõi camera và chất lượng lon theo thời gian thực.</p>
          </div>
          <div className={`processing-pill ${isProcessing ? "is-running" : "is-paused"}`}>
            <span className="processing-pill__signal" />
            <span>{isProcessing ? "Đang kiểm tra" : "Đã tạm dừng"}</span>
          </div>
        </section>

        <section className="metrics-strip" aria-label="Chỉ số sản xuất">
          <div className="yield-metric">
            <div className="yield-metric__copy">
              <span className="metric__label">Tỷ lệ đạt</span>
              <strong className="yield-metric__value">{yieldRate === null ? "—" : `${yieldRate.toFixed(1)}%`}</strong>
            </div>
            <div className="yield-meter" style={{ "--yield": `${yieldRate ?? 0}%` }} aria-label={yieldRate === null ? "Chưa có dữ liệu" : `${yieldRate.toFixed(1)} phần trăm đạt`}>
              <span />
            </div>
          </div>
          <Metric label="Đã kiểm tra" value={loading ? "···" : total} note="tổng số lon" icon={Pulse} index={1} />
          <Metric label="Đạt chuẩn" value={loading ? "···" : status?.good ?? 0} note="GOOD" tone="good" icon={CheckCircle} index={2} />
          <Metric label="Cần loại" value={loading ? "···" : status?.ng ?? 0} note="NG" tone="ng" icon={WarningCircle} index={3} />
          <Metric label="Camera FPS" value={loading ? "···" : status?.fps ?? 0} note="khung hình / giây" icon={Gauge} index={4} />
        </section>

        <section className="monitor-grid" aria-label="Giám sát dây chuyền">
          <article className="feed-panel">
            <div className="section-heading feed-panel__heading">
              <div>
                <p className="eyebrow">VISUAL MONITOR / CAM-01</p>
                <h2>Camera trực tiếp</h2>
              </div>
              <div className="feed-panel__tools">
                <span className="fps-chip"><Gauge size={14} aria-hidden="true" />{status?.fps ?? 0} FPS</span>
                <button className="icon-button" type="button" onClick={() => { setCameraError(false); setStreamKey((key) => key + 1); }} aria-label="Tải lại luồng camera">
                  <ArrowClockwise size={17} aria-hidden="true" />
                </button>
              </div>
            </div>

            <div className={`video-stage ${cameraError ? "has-error" : ""}`}>
              {!cameraError ? (
                <img
                  key={streamKey}
                  src={videoSrc}
                  alt="Luồng camera giám sát lon trên băng chuyền"
                  onLoad={() => { setCameraReady(true); setCameraError(false); }}
                  onError={() => { setCameraReady(false); setCameraError(true); }}
                />
              ) : (
                <div className="video-empty" role="status">
                  <div className="video-empty__icon"><Camera size={25} aria-hidden="true" /></div>
                  <strong>Chưa nhận được hình ảnh</strong>
                  <span>Kiểm tra camera và API backend, sau đó tải lại luồng.</span>
                  <button className="text-action" type="button" onClick={() => { setCameraError(false); setStreamKey((key) => key + 1); }}>Thử kết nối lại <ArrowClockwise size={14} aria-hidden="true" /></button>
                </div>
              )}
              <div className="video-stage__topline" aria-hidden="true">
                <span className={`live-indicator ${cameraReady ? "is-live" : ""}`}><span />{cameraReady ? "LIVE" : "STANDBY"}</span>
                <span className="camera-id">CAM-01&nbsp; / &nbsp;INSPECTION BAY</span>
              </div>
              {!cameraError && <div className="scanline" aria-hidden="true" />}
              <div className="video-stage__corners" aria-hidden="true"><i /><i /><i /><i /></div>
              {status?.last_verdict && (
                <div className={`frame-verdict ${status.last_verdict === "GOOD" ? "is-good" : "is-ng"}`}>
                  LẦN CUỐI <b>{status.last_verdict}</b>
                </div>
              )}
            </div>

            <div className="feed-caption">
              <div className="feed-caption__left"><span className="feed-caption__key">PIPELINE</span><span>Object segmentation <i /> anomaly scoring</span></div>
              <span className="feed-caption__right"><span className="feed-caption__key">STATUS</span>{status?.status_text || "Đang chờ tín hiệu"}</span>
            </div>
          </article>

          <aside className="control-rail">
            <section className="machine-panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">MACHINE STATE</p>
                  <h2>Line 01</h2>
                </div>
                <div className={`machine-icon ${isConnected ? "is-online" : ""}`}><Cpu size={19} aria-hidden="true" /></div>
              </div>
              <div className="machine-state">
                <span className={`machine-state__dot ${isConnected ? "is-online" : "is-offline"}`} />
                <div>
                  <strong>{isConnected ? "Controller online" : "Controller offline"}</strong>
                  <span>{status?.conveyor === "running" ? "Lệnh motor: chạy" : "Lệnh motor: dừng"}</span>
                </div>
              </div>
              <div className="machine-meta">
                <div><span>Inspection</span><strong>{isProcessing ? "ACTIVE" : "PAUSED"}</strong></div>
                <div><span>Last verdict</span><strong className={status?.last_verdict === "NG" ? "text-ng" : "text-good"}>{status?.last_verdict || "—"}</strong></div>
              </div>
              <div className="action-stack">
                <motion.button
                  className={`primary-action ${isProcessing ? "is-pause" : ""}`}
                  type="button"
                  disabled={busy || Boolean(apiError)}
                  whileHover={reduceMotion ? undefined : { y: -1 }}
                  whileTap={reduceMotion ? undefined : { scale: 0.985 }}
                  onClick={() => submitAction(isProcessing ? "/control/stop" : "/control/start", isProcessing ? "Đã tạm dừng kiểm tra." : "Đã bật kiểm tra.")}
                >
                  {isProcessing ? <Pause size={18} weight="fill" aria-hidden="true" /> : <Play size={18} weight="fill" aria-hidden="true" />}
                  {busy ? "Đang cập nhật…" : isProcessing ? "Tạm dừng kiểm tra" : "Bắt đầu kiểm tra"}
                </motion.button>
                <p className="action-hint">Thao tác này bật/tắt phân tích AI; điều khiển cơ cấu vẫn do ESP xử lý.</p>
              </div>
            </section>

            <section id="controls" className="threshold-panel">
              <div className="section-heading threshold-panel__heading">
                <div><p className="eyebrow">MODEL TUNING</p><h2>Ngưỡng phát hiện</h2></div>
                <SlidersHorizontal size={18} className="section-heading__icon" aria-hidden="true" />
              </div>
              <ThresholdControl
                label="PatchCore anomaly threshold"
                value={thresholds.patchcore}
                min={0.05}
                max={0.5}
                step={0.001}
                formatted={thresholds.patchcore.toFixed(4)}
                onChange={(value) => changeThreshold("patchcore", value)}
              />
              <ThresholdControl
                label="YOLO confidence"
                value={thresholds.conf}
                min={0.5}
                max={0.99}
                step={0.01}
                formatted={thresholds.conf.toFixed(2)}
                onChange={(value) => changeThreshold("conf", value)}
              />
              <div className="threshold-note"><Cpu size={14} aria-hidden="true" /><span>Ngưỡng được cập nhật trực tiếp trong backend.</span></div>
            </section>
          </aside>
        </section>

        <section className="lower-grid">
          <article id="inspections" className="history-panel">
            <div className="section-heading history-panel__heading">
              <div><p className="eyebrow">QUALITY LOG</p><h2>Lịch sử kiểm tra</h2></div>
              <span className="history-count">{history.length} / 100 bản ghi</span>
            </div>
            {visibleHistory.length ? (
              <div className="table-wrap">
                <table className="inspection-table">
                  <thead><tr><th>Sản phẩm</th><th>Thời gian</th><th>Kết quả</th><th>Lỗi phát hiện</th><th>Điểm</th></tr></thead>
                  <tbody>
                    {visibleHistory.map((item, index) => {
                      const defects = Array.isArray(item.defects) ? item.defects : [];
                      return (
                        <motion.tr key={`${item.id}-${item.timestamp}-${index}`} initial={reduceMotion ? false : { opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: index * 0.025 }}>
                          <td><span className="product-id">#{String(item.id).padStart(3, "0")}</span></td>
                          <td className="mono-cell">{item.timestamp || "—"}</td>
                          <td><Verdict value={item.verdict} /></td>
                          <td>{defects.length ? <span className="defect-list">{defects.slice(0, 2).join(", ")}{defects.length > 2 ? ` +${defects.length - 2}` : ""}</span> : <span className="muted-cell">Không phát hiện</span>}</td>
                          <td className="score-cell">{Number.isFinite(Number(item.score)) ? Number(item.score).toFixed(3) : "—"}</td>
                        </motion.tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-log">
                <div className="empty-log__mark"><ClockCounterClockwise size={22} aria-hidden="true" /></div>
                <div><strong>Chưa có kết quả kiểm tra</strong><span>Bản ghi sẽ xuất hiện ở đây sau lon đầu tiên đi qua camera.</span></div>
                <ArrowSquareOut size={16} className="empty-log__arrow" aria-hidden="true" />
              </div>
            )}
          </article>

          <article className="sequence-panel">
            <div className="section-heading">
              <div><p className="eyebrow">RECENT SIGNAL</p><h2>Chuỗi kết quả</h2></div>
              <Pulse size={18} className="section-heading__icon" aria-hidden="true" />
            </div>
            <ResultRail history={history} />
            <div className="sequence-legend"><span><i className="legend-dot is-good" />GOOD</span><span><i className="legend-dot is-ng" />NG</span></div>
            <div className="sequence-footer"><span>Backend refresh</span><strong>{updateLabel}</strong></div>
          </article>
        </section>

        <AnimatePresence>
          {actionMessage && (
            <motion.div className={`action-toast ${actionMessage.startsWith("error:") ? "is-error" : ""}`} role="status" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}>
              {actionMessage.startsWith("error:") ? <WarningCircle size={17} aria-hidden="true" /> : <CheckCircle size={17} aria-hidden="true" />}
              <span>{actionMessage.slice(actionMessage.indexOf(":") + 1)}</span>
              <button type="button" className="icon-button" aria-label="Đóng thông báo" onClick={() => setActionMessage("")}><X size={15} aria-hidden="true" /></button>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {showResetConfirm && (
            <motion.div className="reset-confirm" role="dialog" aria-modal="false" aria-label="Xác nhận xóa thống kê" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}>
              <span>Xóa bộ đếm và lịch sử kiểm tra?</span>
              <button type="button" className="reset-confirm__cancel" onClick={() => setShowResetConfirm(false)}>Hủy</button>
              <button type="button" className="reset-confirm__accept" onClick={async () => { await submitAction("/reset", "Đã đặt lại thống kê."); setHistory([]); setShowResetConfirm(false); }}>Xóa dữ liệu</button>
            </motion.div>
          )}
        </AnimatePresence>

        <footer className="app-footer">
          <span>BEERCAN / VISION</span>
          <span><span className="footer-status-dot" />{apiError ? "Đang chờ backend" : "Backend polling 1.6s"}</span>
          <button type="button" className="footer-reset" onClick={() => setShowResetConfirm((show) => !show)} aria-expanded={showResetConfirm}>
            <ArrowClockwise size={14} aria-hidden="true" /> Đặt lại thống kê
          </button>
        </footer>
      </main>
    </div>
  );
}

function ThresholdControl({ label, value, min, max, step, formatted, onChange }) {
  const ratio = ((value - min) / (max - min)) * 100;
  return (
    <label className="threshold-control">
      <span className="threshold-control__row"><span>{label}</span><output>{formatted}</output></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(event.target.value)} style={{ "--range-progress": `${ratio}%` }} />
      <span className="threshold-control__limits"><span>{min}</span><span>{max}</span></span>
    </label>
  );
}

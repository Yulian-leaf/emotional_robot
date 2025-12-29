import React from 'react';

function EmotionList({ emotion }) {
  if (!Array.isArray(emotion) || emotion.length === 0) return <span className="muted">（无）</span>;
  const top = [...emotion]
    .filter((e) => e && e.label)
    .sort((a, b) => (b.score || 0) - (a.score || 0))
    .slice(0, 3);
  return (
    <span>
      {top.map((e, idx) => (
        <span key={`${e.label}-${idx}`} className="badge" style={{ marginRight: 6 }}>
          {String(e.label)} {typeof e.score === 'number' ? e.score.toFixed(3) : String(e.score || '')}
        </span>
      ))}
    </span>
  );
}

function TimelineTable({ title, timeline }) {
  const rows = Array.isArray(timeline) ? timeline : [];
  return (
    <div>
      <div className="h3">{title}</div>
      {rows.length === 0 ? (
        <div className="muted">（该模态没有时间轴输出）</div>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th style={{ width: 150 }}>start / frame</th>
              <th style={{ width: 110 }}>end</th>
              <th style={{ width: 280 }}>top emotions</th>
              <th>text / summary</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((seg, i) => (
              <tr key={i}>
                <td>{seg?.start ?? seg?.frame ?? ''}</td>
                <td>{seg?.end ?? ''}</td>
                <td><EmotionList emotion={seg?.emotion} /></td>
                <td>
                  {seg?.text
                    ? String(seg.text)
                    : seg?.summary
                      ? String(seg.summary)
                      : <span className="muted">—</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function Timeline({ data }) {
  const summary = (data && data.summary) || '';

  const rawTimeline = data && data.timeline;
  let tl = {};
  if (Array.isArray(rawTimeline)) {
    const first = rawTimeline[0] || {};
    if (first && typeof first === 'object' && 'frame' in first) {
      tl = { image: rawTimeline };
    } else if (first && typeof first === 'object' && 'text' in first) {
      tl = { text: rawTimeline };
    } else {
      tl = { audio: rawTimeline };
    }
  } else {
    tl = rawTimeline || {};
  }

  const transcript =
    (data && (data.transcript || (data.modalities && data.modalities.text && data.modalities.text.timeline && data.modalities.text.timeline[0] && data.modalities.text.timeline[0].text))) ||
    (Array.isArray(tl.text) && tl.text[0] && tl.text[0].text) ||
    '';

  const textDebug = (data && data.modalities && data.modalities.text && data.modalities.text._debug) || (data && data._debug) || null;

  return (
    <div>
      <div className="h2">结果</div>

      <div className="kv">
        <div className="label">融合结论</div>
        <div>{summary ? <span className="badge">{String(summary)}</span> : <span className="muted">（无）</span>}</div>

        <div className="label">转写文本</div>
        <div>
          {transcript ? (
            <pre className="pre" style={{ background: '#111827' }}>{transcript}</pre>
          ) : (
            <div>
              <span className="muted">（暂无：未启用 ASR 或转写失败）</span>
              {textDebug ? (
                <div style={{ marginTop: 8 }}>
                  <div className="muted">_debug：</div>
                  <pre className="pre">{JSON.stringify(textDebug, null, 2)}</pre>
                </div>
              ) : null}
            </div>
          )}
        </div>
      </div>

      <TimelineTable title="文本情感（text）" timeline={tl.text} />
      <TimelineTable title="语音情感（audio）" timeline={tl.audio} />
      <TimelineTable title="表情情感（image）" timeline={tl.image} />
    </div>
  );
}

export default Timeline;

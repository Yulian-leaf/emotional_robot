import React, { useState } from 'react';
import axios from 'axios';
import VideoPlayer from './components/VideoPlayer';
import Timeline from './components/Timeline';

function App() {
  const [video, setVideo] = useState(null);
  const [result, setResult] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState('');
  const [queryId, setQueryId] = useState('');
  const [error, setError] = useState('');
  const [isPolling, setIsPolling] = useState(false);

  const [textInput, setTextInput] = useState('');
  const [textResult, setTextResult] = useState(null);
  const [textError, setTextError] = useState('');
  const [textBusy, setTextBusy] = useState(false);

  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState('');
  const [imageResult, setImageResult] = useState(null);
  const [imageError, setImageError] = useState('');
  const [imageBusy, setImageBusy] = useState(false);

  const handleUpload = async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    setError('');
    setResult(null);
    setStatus('uploading');
    setVideo(URL.createObjectURL(file));

    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post('/api/predict/video', formData);
      const id = res.data && res.data.task_id;
      if (!id) throw new Error('后端未返回 task_id');
      setTaskId(id);
      setQueryId(id);
      setStatus('processing');
      pollStatus(id);
    } catch (err) {
      setStatus('error');
      setError(err?.response?.data?.detail || err?.message || '上传失败');
    }
  };

  const pollStatus = async (id) => {
    if (!id) return;
    if (isPolling) return;

    setIsPolling(true);
    setError('');

    const tick = async () => {
      try {
        const res = await axios.get(`/api/predict/status/${id}`);
        const st = res?.data?.status;
        setStatus(st || 'processing');
        if (st === 'done') {
          const resultRes = await axios.get(`/api/predict/result/${id}`);
          setResult(resultRes.data);
          setIsPolling(false);
          return;
        }
        if (st === 'error') {
          setError(res?.data?.error || '后端任务失败');
          setIsPolling(false);
          return;
        }
        if (st === 'not_found') {
          setError('任务不存在：请检查任务ID');
          setIsPolling(false);
          return;
        }
      } catch (err) {
        setError(err?.response?.data?.detail || err?.message || '查询状态失败');
        setIsPolling(false);
        return;
      }

      setTimeout(tick, 1500);
    };

    tick();
  };

  const handleQuery = async () => {
    if (!queryId.trim()) return;
    setTaskId(queryId.trim());
    setResult(null);
    setStatus('processing');
    pollStatus(queryId.trim());
  };

  const copyTaskId = async () => {
    if (!taskId) return;
    try {
      await navigator.clipboard.writeText(taskId);
    } catch {
      // ignore
    }
  };

  const runTextEmotion = async () => {
    const text = textInput.trim();
    if (!text) return;
    setTextBusy(true);
    setTextError('');
    setTextResult(null);
    try {
      const res = await axios.post('/api/predict/text', { text });
      setTextResult(res.data);
    } catch (err) {
      setTextError(err?.response?.data?.detail || err?.message || '文本情绪识别失败');
    } finally {
      setTextBusy(false);
    }
  };

  const onPickImage = async (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    setImageFile(f);
    setImagePreview(URL.createObjectURL(f));
    setImageResult(null);
    setImageError('');
  };

  const runImageEmotion = async () => {
    if (!imageFile) return;
    setImageBusy(true);
    setImageError('');
    setImageResult(null);
    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      const res = await axios.post('/api/predict/image', formData);
      setImageResult(res.data);
    } catch (err) {
      setImageError(err?.response?.data?.detail || err?.message || '图片情绪识别失败');
    } finally {
      setImageBusy(false);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <h1 className="h1">多模态情感识别</h1>
        <div className="muted">上传视频 → 后台处理 → 查询任务状态/结果</div>
      </div>

      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div className="row">
            <span className="label">选择视频：</span>
            <input type="file" accept="video/*" onChange={handleUpload} />
          </div>
          <div className="badge">状态：{status || 'idle'}</div>
        </div>

        {taskId && (
          <div style={{ marginTop: 10 }} className="kv">
            <div className="label">任务ID</div>
            <div className="row">
              <div className="badge" title={taskId}>{taskId}</div>
              <button className="button" onClick={copyTaskId}>复制</button>
            </div>
          </div>
        )}

        <div style={{ marginTop: 12 }} className="row">
          <span className="label">查询任务：</span>
          <input
            className="input"
            value={queryId}
            onChange={(e) => setQueryId(e.target.value)}
            placeholder="粘贴 task_id 后查询"
          />
          <button className="button" onClick={handleQuery} disabled={!queryId.trim() || isPolling}>查询</button>
        </div>

        {error && <div style={{ marginTop: 10 }} className="error">{String(error)}</div>}
      </div>

      {video && (
        <div className="card">
          <div className="h2">预览</div>
          <VideoPlayer src={video} />
        </div>
      )}

      {result && (
        <div className="card">
          <Timeline data={result} />
        </div>
      )}

      <div className="card">
        <div className="h2">文本情绪识别</div>
        <div className="muted">直接输入文本，调用 /api/predict/text</div>
        <div style={{ marginTop: 10 }}>
          <textarea
            className="input"
            style={{ width: '100%', minHeight: 90 }}
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="输入一句话，例如：I am very happy today!"
          />
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <button className="button" onClick={runTextEmotion} disabled={textBusy || !textInput.trim()}>
            {textBusy ? '识别中…' : '开始识别'}
          </button>
          {textError && <span className="error">{String(textError)}</span>}
        </div>
        {textResult && (
          <div style={{ marginTop: 12 }}>
            <Timeline data={textResult} />
          </div>
        )}
      </div>

      <div className="card">
        <div className="h2">图片情绪识别</div>
        <div className="muted">上传单张图片，调用 /api/predict/image</div>
        <div className="row" style={{ marginTop: 10, justifyContent: 'space-between' }}>
          <div className="row">
            <span className="label">选择图片：</span>
            <input type="file" accept="image/*" onChange={onPickImage} />
          </div>
          <button className="button" onClick={runImageEmotion} disabled={imageBusy || !imageFile}>
            {imageBusy ? '识别中…' : '开始识别'}
          </button>
        </div>
        {imageError && <div style={{ marginTop: 10 }} className="error">{String(imageError)}</div>}
        {imagePreview && (
          <div style={{ marginTop: 10 }}>
            <img
              src={imagePreview}
              alt="preview"
              style={{ maxWidth: '100%', borderRadius: 10, border: '1px solid #e5e7eb' }}
            />
          </div>
        )}
        {imageResult && (
          <div style={{ marginTop: 12 }}>
            <Timeline data={imageResult} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

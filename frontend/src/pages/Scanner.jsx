import { useEffect, useRef, useState } from "react";
import { AlertCircle, ArrowLeft, ArrowRight, Camera, Check, ImagePlus, LoaderCircle, RotateCcw, ScanLine, ShieldCheck, Upload, X } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import PageHeader from "../components/PageHeader.jsx";
import { analyzeLeaf } from "../services/api.js";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_FILE_BYTES = 8 * 1024 * 1024;

export default function Scanner() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [dragging, setDragging] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [apiError, setApiError] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [progressStep, setProgressStep] = useState(0);
  const inputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!cameraOn) return undefined;
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Camera access requires a secure browser context. Use HTTPS or localhost, or upload a photo instead.");
      setCameraOn(false);
      return undefined;
    }
    let active = true;
    navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false })
      .then((stream) => {
        if (!active) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch((error) => {
        setCameraError(error.name === "NotAllowedError"
          ? "Camera access was denied. Allow camera permission in your browser settings, or upload a photo instead."
          : "We couldn’t start the camera. Check that your device has a camera, or upload a photo instead.");
        setCameraOn(false);
      });
    return () => {
      active = false;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, [cameraOn]);

  useEffect(() => () => {
    if (preview.startsWith("blob:")) URL.revokeObjectURL(preview);
  }, [preview]);

  function chooseFile(nextFile) {
    setApiError("");
    setCameraError("");
    if (!nextFile) return;
    if (!ACCEPTED_TYPES.includes(nextFile.type)) {
      setApiError("Choose a JPEG, PNG, or WebP image.");
      return;
    }
    if (nextFile.size > MAX_FILE_BYTES) {
      setApiError("This image is larger than 8 MB. Choose a smaller image.");
      return;
    }
    if (preview.startsWith("blob:")) URL.revokeObjectURL(preview);
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
    setCameraOn(false);
  }

  function removeImage() {
    if (preview.startsWith("blob:")) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview("");
    setApiError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  function capturePhoto() {
    const video = videoRef.current;
    if (!video?.videoWidth) {
      setCameraError("The camera is still starting. Try capturing again in a moment.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) {
        setCameraError("The camera photo could not be captured. Please try again.");
        return;
      }
      chooseFile(new File([blob], `apple-leaf-${Date.now()}.jpg`, { type: "image/jpeg" }));
    }, "image/jpeg", 0.92);
  }

  async function submitAnalysis() {
    if (!file || analyzing) return;
    setAnalyzing(true);
    setApiError("");
    setProgressStep(0);
    const stepTimer = window.setInterval(() => setProgressStep((step) => Math.min(step + 1, 2)), 1500);
    try {
      const { scan } = await analyzeLeaf(file);
      navigate(`/analysis/${scan.id}`, { state: { scan } });
    } catch (error) {
      setApiError(error.message);
    } finally {
      window.clearInterval(stepTimer);
      setAnalyzing(false);
    }
  }

  return (
    <div>
      <PageHeader eyebrow="AI LEAF SCANNER" title="Take a closer look." description="A local MobileNetV2 model ranks four classes from your photo. It does not identify symptoms, causes, or treatment, and cannot confirm the image is an apple leaf." />
      <div className="scanner-layout">
        <section className="panel scanner-panel">
          <div className="scanner-panel-head"><span className="scanner-step">01</span><div><h2>Add a leaf photo</h2><p>Use a clear, close-up image in natural light.</p></div></div>
          {cameraOn ? <div className="camera-view"><video ref={videoRef} autoPlay playsInline muted /><div className="camera-overlay"><span>Position one leaf in the frame</span></div><div className="camera-controls"><button className="button button-white" onClick={() => setCameraOn(false)}><X size={16} /> Cancel</button><button className="capture-button" onClick={capturePhoto} aria-label="Capture leaf photo"><span /></button><span className="camera-hint">Tap to capture</span></div></div>
            : preview ? <div className="preview-stage"><img src={preview} alt="Selected apple leaf to analyze" /><button className="remove-photo" onClick={removeImage}><X size={15} /> Remove photo</button></div>
              : <div className={`upload-area ${dragging ? "upload-area-dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files[0]); }}>
                <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => chooseFile(event.target.files[0])} hidden />
                <div className="upload-illustration"><ImagePlus size={23} /></div><strong>Drop your leaf photo here</strong><span>or choose from your device</span>
                <div className="upload-buttons"><button className="button button-dark" onClick={() => inputRef.current?.click()}><Upload size={16} /> Browse files</button><button className="button button-outline" onClick={() => { setCameraError(""); setCameraOn(true); }}><Camera size={16} /> Use camera</button></div>
                <small>JPEG, PNG or WebP · Up to 8 MB</small>
              </div>}
          {cameraError && <div className="notice notice-error camera-notice"><AlertCircle size={16} />{cameraError}</div>}
          {apiError && <div className="notice notice-error"><AlertCircle size={16} />{apiError}</div>}
          <div className="scanner-submit-row"><span><ShieldCheck size={16} /> Photo processed by the local model and saved to scan history.</span><button className="button button-green" disabled={!file || analyzing} onClick={submitAnalysis}>{analyzing ? <><LoaderCircle className="spin" size={17} /> Analyzing…</> : <><ScanLine size={17} /> Analyze leaf <ArrowRight size={15} /></>}</button></div>
          {analyzing && <div className="analysis-progress"><div className="progress-track"><span /></div><div className="progress-labels"><span className={progressStep >= 0 ? "progress-active" : ""}><Check size={12} /> Image received</span><span className={progressStep >= 1 ? "progress-active" : ""}><Check size={12} /> Running MobileNetV2</span><span className={progressStep >= 2 ? "progress-active" : ""}><Check size={12} /> Preparing class scores</span></div></div>}
        </section>
        <aside className="scanner-aside">
          <div className="panel photo-tips"><div className="eyebrow">FOR A CLEARER LOOK</div><h3>A few photo tips</h3><ul><li><span>01</span><div><strong>Find soft daylight</strong><small>Avoid harsh shadows and direct glare.</small></div></li><li><span>02</span><div><strong>Focus on one leaf</strong><small>Make symptoms easy to see in the frame.</small></div></li><li><span>03</span><div><strong>Keep the image steady</strong><small>Hold still so small details stay sharp.</small></div></li></ul></div>
          <div className="scanner-disclaimer"><span><ShieldCheck size={18} /></span><div><strong>Prototype screening only</strong><p>The supplied model scored 55.6% accuracy on 63 held-out images. Its score is not calibrated; confirm results with a local expert before making treatment decisions.</p></div></div>
          <Link className="scanner-library-link" to="/diseases"><ArrowLeft size={14} /> Learn about apple leaf conditions</Link>
        </aside>
      </div>
    </div>
  );
}

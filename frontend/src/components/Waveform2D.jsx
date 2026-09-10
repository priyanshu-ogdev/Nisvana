/**
 * Waveform2D.jsx — F-4: Dual-stream waveform visualizer.
 * 
 * Shows either "enhanced" (post-model) or "raw" (pre-model) FFT bins.
 * Click toggles between the two streams. A small badge shows which is active.
 * Both streams come from fftStreams[clientId].{enhanced, raw}.
 */
import { useEffect, useRef, useState } from 'react';
import { fftStreams } from '../state/useConnectionStore';

export function Waveform2D({ clientId, isMuted, isSecure }) {
  const canvasRef = useRef(null);
  // F-4: toggle between 'enhanced' and 'raw'
  const [stream, setStream] = useState('enhanced');

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    let animationFrameId;
    
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    
    const count = 64;
    const barWidth = (width / count) - 1.5;
    const gap = 1.5;

    const render = () => {
      ctx.clearRect(0, 0, width, height);
      
      // F-4: read the active stream from the dual-stream object
      const streamData = fftStreams[clientId];
      const bins = streamData[stream] || streamData.enhanced;
      const isRaw = stream === 'raw';

      for (let i = 0; i < count; i++) {
        let val = 0;
        let c;
        
        if (isSecure && !isMuted) {
           val = (bins[i] || 0) / 255;
           ctx.shadowBlur = val > 0.1 ? 6 : 0;
           if (isRaw) {
             // Raw stream: warmer, amber-ish color indicating unprocessed signal
             ctx.shadowColor = '#C97B8A';
             c = `rgba(201, 123, 138, ${0.4 + val * 0.6})`;
           } else {
             // Enhanced stream: teal/cyan indicating processed signal
             ctx.shadowColor = '#43D7E5';
             c = `rgba(67, 215, 229, ${0.5 + val * 0.5})`;
           }
        } else {
           ctx.shadowBlur = 0;
           c = isMuted ? '#C97B8A' : '#52657D';
        }
        
        const h = Math.max(2, val * (height - 4));
        const x = i * (barWidth + gap);
        const y = height - h;
        
        ctx.fillStyle = c;
        ctx.fillRect(x, y, barWidth, h);
      }
      
      animationFrameId = requestAnimationFrame(render);
    };
    
    render();
    
    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [clientId, isMuted, isSecure, stream]);

  return (
    <div className="relative w-full h-full">
      <canvas 
        ref={canvasRef} 
        className="w-full h-full block cursor-pointer" 
        onClick={() => isSecure && setStream(s => s === 'enhanced' ? 'raw' : 'enhanced')}
        title={`Showing ${stream} stream. Click to toggle.`}
      />
      {/* Stream badge */}
      {isSecure && (
        <div
          className={`absolute top-1 right-1 text-[7px] font-bold tracking-widest px-1 py-0.5 rounded pointer-events-none
            ${stream === 'enhanced' ? 'bg-[var(--cyan)]/20 text-[var(--cyan)]' : 'bg-[var(--rose)]/20 text-[var(--rose)]'}`}
        >
          {stream === 'enhanced' ? 'ENC' : 'RAW'}
        </div>
      )}
    </div>
  );
}

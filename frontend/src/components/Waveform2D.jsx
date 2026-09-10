import { useEffect, useRef } from 'react';
import { fftStreams } from '../state/useConnectionStore';

export function Waveform2D({ clientId, isMuted, isSecure }) {
  const canvasRef = useRef(null);

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

    const slate = '#52657D'; // --text-low
    const mutedRose = '#C97B8A';

    const render = () => {
      ctx.clearRect(0, 0, width, height);
      
      const bins = fftStreams[clientId];
      
      for (let i = 0; i < count; i++) {
        let val = 0;
        let c = isMuted ? mutedRose : slate;
        
        if (isSecure && !isMuted) {
           val = (bins[i] || 0) / 255;
           ctx.shadowBlur = val > 0.1 ? 6 : 0;
           ctx.shadowColor = '#43D7E5'; // --cyan
           c = `rgba(67, 215, 229, ${0.5 + val * 0.5})`; // teal/cyan mix
        } else {
           ctx.shadowBlur = 0;
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
  }, [clientId, isMuted, isSecure]);

  return (
    <canvas 
      ref={canvasRef} 
      className="w-full h-full block" 
    />
  );
}

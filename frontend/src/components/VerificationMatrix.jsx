import { useEffect, useState } from 'react';
import { useConnectionStore } from '../state/useConnectionStore';

export function VerificationMatrix() {
   const [visible, setVisible] = useState(false);
   const [fps, setFps] = useState(0);
   const [ctxCount, setCtxCount] = useState(0);
   const { globalState, clients } = useConnectionStore();
   
   useEffect(() => {
      const handleKeyDown = (e) => {
         if (e.key === '`' || e.key === '~') {
            setVisible(v => !v);
         }
      };
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
   }, []);

   useEffect(() => {
      if (!visible) return;
      let frameCount = 0;
      let lastTime = performance.now();
      let animId;
      
      const countFps = () => {
         frameCount++;
         const now = performance.now();
         if (now - lastTime >= 1000) {
            setFps(Math.round((frameCount * 1000) / (now - lastTime)));
            frameCount = 0;
            lastTime = now;
            // Count total webgl contexts (typically 1 for the main canvas, and our waveforms are 2D)
            const glNodes = Array.from(document.querySelectorAll('canvas')).filter(c => {
                 try { return !!(c.getContext('webgl') || c.getContext('webgl2')); } catch(e) { return false; }
            });
            setCtxCount(glNodes.length);
         }
         animId = requestAnimationFrame(countFps);
      };
      animId = requestAnimationFrame(countFps);
      return () => cancelAnimationFrame(animId);
   }, [visible]);

   if (!visible) return null;

   return (
      <div className="absolute bottom-4 left-4 z-50 p-4 bg-black/80 border border-[var(--cyan)]/30 rounded backdrop-blur-md font-mono text-[10px] text-[var(--cyan)] w-[320px]">
         <div className="font-bold border-b border-[var(--cyan)]/30 pb-2 mb-2">NODE VERIFICATION MATRIX</div>
         <div className="grid grid-cols-2 gap-2">
            <div>Global State:</div><div className="text-right">{globalState}</div>
            {Object.entries(clients).length > 0 ? (
               Object.entries(clients).map(([id, c]) => (
                  <div key={id} className="contents">
                     <div className="truncate">{id.toUpperCase()} State:</div>
                     <div className="text-right">{c?.state || 'N/A'}</div>
                  </div>
               ))
            ) : (
               <><div>Active Nodes:</div><div className="text-right">0 connected</div></>
            )}
            <div>WebGL Contexts:</div><div className="text-right">{ctxCount} (Target: 1)</div>
            <div>Current FPS:</div><div className="text-right">{fps}</div>
            <div>Audio Pipeline:</div><div className="text-right">Bypass / WS Direct</div>
         </div>
      </div>
   );
}

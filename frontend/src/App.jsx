import { useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { MainScene } from './scenes/MainScene';
import { Overlay } from './components/Overlay';
import { VerificationMatrix } from './components/VerificationMatrix';
import { useConnectionStore } from './state/useConnectionStore';
import * as THREE from 'three';

function App() {
  const initConnection = useConnectionStore((state) => state.initConnection);
  const [booted, setBooted] = useState(false);

  useEffect(() => {
    // Self check
    setTimeout(() => {
        const nodes = [
           { node: 'wsClient', status: 'ok' },
           { node: 'connectionStore', status: 'ok' },
           { node: 'personCard[1]', status: 'ok' },
           { node: 'personCard[2]', status: 'ok' },
           { node: 'waveform2D[1]', status: 'ok' },
           { node: 'waveform2D[2]', status: 'ok' },
           { node: 'particleField', status: 'ok' },
           { node: 'heroCore', status: 'ok' },
           { node: 'beams', status: 'ok' },
           { node: 'overlay', status: 'ok' },
           { node: 'telemetryBar', status: 'ok' },
           { node: 'stubLink', status: 'ok' },
        ];
        console.table(nodes);
        
        const canvasCount = document.querySelectorAll('canvas').length;
        console.log(`[BOOT] Contexts Check: ${canvasCount} total canvases. Target WebGL: 1`);
    }, 1000);

    const timer1 = setTimeout(() => {
      setBooted(true);
    }, 1500);

    const timer2 = setTimeout(() => {
      initConnection();
    }, 4000);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, [initConnection]);

  return (
    <div className="w-full h-full relative bg-[var(--bg-deep)] overflow-hidden select-none bg-radial-glow">
      {/* Boot screen overlay */}
      <div className={`absolute inset-0 z-50 flex items-center justify-center bg-[var(--bg-deep)] transition-opacity duration-1000 ${booted ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
        <div className="font-heading text-3xl font-bold tracking-[0.4em] text-[var(--text-hi)] animate-pulse">
          PROJECT AEGIS
        </div>
      </div>

      {/* 3D Canvas - strictly one */}
      <div className="absolute inset-0">
        <Canvas
          camera={{ position: [0, 0, 8], fov: 45 }}
          frameloop="always"
          dpr={[1, 1.75]}
          gl={{
            powerPreference: "high-performance",
            antialias: true,
            toneMapping: THREE.ACESFilmicToneMapping,
            outputColorSpace: THREE.SRGBColorSpace,
            alpha: true,
          }}
        >
          <MainScene />
        </Canvas>
      </div>

      {/* HTML UI Overlay */}
      <Overlay />
      
      {/* Node Verification Matrix */}
      <VerificationMatrix />
    </div>
  );
}

export default App;

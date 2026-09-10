import { ParticleField } from '../components/ParticleField';
import { CentralCore } from '../components/CentralCore';
import { HandshakeBeams } from '../components/HandshakeBeams';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';

export function MainScene() {
  return (
    <>
      <ambientLight intensity={0.5} />
      
      <ParticleField />
      
      <group scale={[1.8, 1.8, 1.8]}>
         <CentralCore />
      </group>

      <HandshakeBeams />

      <EffectComposer disableNormalPass>
        <Bloom 
          luminanceThreshold={0.18} 
          mipmapBlur 
          intensity={1.15} 
        />
        <Vignette offset={0.28} darkness={0.55} />
      </EffectComposer>
    </>
  );
}

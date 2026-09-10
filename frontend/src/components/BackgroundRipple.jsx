import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const vertexShader = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fragmentShader = `
uniform float uTime;
uniform float uAmplitude;
varying vec2 vUv;

void main() {
  vec2 uv = vUv * 2.0 - 1.0;
  // Account for aspect ratio if needed, but simple ripple is fine
  float dist = length(uv);
  
  // Create ripple rings
  float ripple = sin(dist * 40.0 - uTime * 1.5);
  // Fade out ripples towards the edge
  ripple *= exp(-dist * 3.0);
  
  vec3 baseColor = vec3(0.02, 0.024, 0.039); // #05060A
  vec3 rippleColor = vec3(0.31, 0.95, 0.84) * 0.08 * (uAmplitude + 0.1); // subtle cyan glow
  
  // Smoothstep to soften the ripple
  ripple = smoothstep(0.0, 1.0, ripple);
  
  vec3 finalColor = baseColor + ripple * rippleColor;
  
  gl_FragColor = vec4(finalColor, 1.0);
}
`;

export function BackgroundRipple({ getGlobalAmplitude }) {
  const materialRef = useRef();

  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
      const targetAmp = getGlobalAmplitude ? getGlobalAmplitude() : 0;
      materialRef.current.uniforms.uAmplitude.value = THREE.MathUtils.lerp(
        materialRef.current.uniforms.uAmplitude.value,
        targetAmp,
        0.05
      );
    }
  });

  return (
    <mesh position={[0, 0, -10]}>
      <planeGeometry args={[100, 50]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={{
          uTime: { value: 0 },
          uAmplitude: { value: 0 },
        }}
        depthWrite={false}
      />
    </mesh>
  );
}

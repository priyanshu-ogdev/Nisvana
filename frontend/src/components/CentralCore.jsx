import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export function CentralCore() {
  const innerRef = useRef();
  const wireRef = useRef();
  const ringRef = useRef();
  
  const violet = new THREE.Color('#8B7CF6');
  const cyan = new THREE.Color('#43D7E5');
  const teal = new THREE.Color('#00AFA5');

  useFrame((state, delta) => {
    const t = state.clock.elapsedTime;
    
    if (wireRef.current) {
       wireRef.current.rotation.x += 0.1 * delta;
       wireRef.current.rotation.y += 0.15 * delta;
    }
    
    if (ringRef.current) {
       ringRef.current.rotation.z -= 0.2 * delta;
       ringRef.current.rotation.x = Math.PI / 2 + Math.sin(t * 0.5) * 0.2;
    }
    
    if (innerRef.current) {
       innerRef.current.rotation.y -= 0.1 * delta;
       const breathe = Math.sin(t * Math.PI * 0.4);
       innerRef.current.material.emissiveIntensity = 2.0 + breathe * 0.2;
    }
  });

  return (
    <group>
      {/* Inner Emissive Sphere */}
      <mesh ref={innerRef}>
        <icosahedronGeometry args={[0.9, 2]} />
        <meshStandardMaterial 
          color="#000000"
          emissive={cyan}
          emissiveIntensity={2.0}
          wireframe={false}
        />
      </mesh>

      {/* Wireframe Icosahedron */}
      <mesh ref={wireRef}>
        <icosahedronGeometry args={[1.2, 1]} />
        <meshBasicMaterial 
          color={violet}
          wireframe={true}
          transparent
          opacity={0.8}
        />
      </mesh>

      {/* Slow-rotating Ring */}
      <mesh ref={ringRef}>
        <torusGeometry args={[1.6, 0.015, 8, 64]} />
        <meshBasicMaterial color={teal} />
      </mesh>
    </group>
  );
}

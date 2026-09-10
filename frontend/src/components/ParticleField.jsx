import { useRef, useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

export function ParticleField() {
  const count = 1000;
  const meshRef = useRef();
  
  const homeRef = useRef(new Float32Array(count * 3));
  const posRef = useRef(new Float32Array(count * 3));
  const velRef = useRef(new Float32Array(count * 3));
  const alphasRef = useRef(new Float32Array(count));
  
  const { viewport } = useThree();

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    const teal = new THREE.Color('#00AFA5'); 
    const cyan = new THREE.Color('#43D7E5'); 
    
    const hw = viewport.width / 2 * 1.5;
    const hh = viewport.height / 2 * 1.5;

    for (let i = 0; i < count; i++) {
       const x = (Math.random() - 0.5) * hw * 2;
       const y = (Math.random() - 0.5) * hh * 2;
       const z = 0;
       
       positions[i*3] = x;
       positions[i*3+1] = y;
       positions[i*3+2] = z;
       
       homeRef.current[i*3] = x;
       homeRef.current[i*3+1] = y;
       homeRef.current[i*3+2] = z;
       
       posRef.current[i*3] = x;
       posRef.current[i*3+1] = y;
       posRef.current[i*3+2] = z;
       
       velRef.current[i*3] = 0;
       velRef.current[i*3+1] = 0;
       velRef.current[i*3+2] = 0;
       
       const c = new THREE.Color().lerpColors(teal, cyan, Math.random());
       colors[i*3] = c.r;
       colors[i*3+1] = c.g;
       colors[i*3+2] = c.b;
       
       alphasRef.current[i] = 0.55;
    }
    
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setAttribute('alpha', new THREE.BufferAttribute(alphasRef.current, 1));
    return geo;
  }, [viewport]);

  const R = 1.5;
  const F_MAX = 18;
  const K = 26;
  const D = 5.5;

  useFrame((state, delta) => {
     if (!meshRef.current) return;
     const dt = Math.min(delta, 1/30);
     
     const px = (state.pointer.x * viewport.width) / 2;
     const py = (state.pointer.y * viewport.height) / 2;

     const positions = geometry.attributes.position.array;
     const alphas = geometry.attributes.alpha.array;
     
     const home = homeRef.current;
     const pos = posRef.current;
     const vel = velRef.current;
     
     let needsUpdate = false;
     let sw = window._aegisShockwave;

     for (let i = 0; i < count; i++) {
        const i3 = i * 3;
        
        let vx = vel[i3];
        let vy = vel[i3+1];
        
        const hx = home[i3];
        const hy = home[i3+1];
        
        let x = pos[i3];
        let y = pos[i3+1];
        
        const dx = x - px;
        const dy = y - py;
        const dist = Math.sqrt(dx*dx + dy*dy);
        
        if (dist < R && dist > 0.001) {
           const force = F_MAX * Math.pow(1 - dist/R, 2);
           vx += (dx / dist) * force * dt;
           vy += (dy / dist) * force * dt;
        }
        
        if (sw) {
             const dsx = x - sw.x;
             const dsy = y - sw.y;
             const sdist = Math.sqrt(dsx*dsx + dsy*dsy);
             const sforce = sw.strength / (1 + sdist * 1.5);
             if (sdist > 0.01 && sdist < 8) {
                 vx += (dsx / sdist) * sforce;
                 vy += (dsy / sdist) * sforce;
             }
        }

        const ax = (hx - x) * K - vx * D;
        const ay = (hy - y) * K - vy * D;
        
        vx += ax * dt;
        vy += ay * dt;
        
        x += vx * dt;
        y += vy * dt;
        
        vel[i3] = vx;
        vel[i3+1] = vy;
        pos[i3] = x;
        pos[i3+1] = y;
        
        positions[i3] = x;
        positions[i3+1] = y;
        
        const speed = Math.sqrt(vx*vx + vy*vy);
        const op = 0.55 + Math.min(0.45, speed * 0.6);
        
        if (speed > 0.01 || Math.abs(x - hx) > 0.01 || alphas[i] !== 0.55) {
            alphas[i] = op;
            needsUpdate = true;
        } else {
            pos[i3] = hx;
            pos[i3+1] = hy;
            positions[i3] = hx;
            positions[i3+1] = hy;
            vel[i3] = 0;
            vel[i3+1] = 0;
            alphas[i] = 0.55;
        }
     }
     
     if (sw) {
         window._aegisShockwave = null;
     }

     if (needsUpdate) {
        geometry.attributes.position.needsUpdate = true;
        geometry.attributes.alpha.needsUpdate = true;
     }
  });

  return (
    <points ref={meshRef} geometry={geometry}>
      <shaderMaterial 
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        vertexShader={`
          attribute float alpha;
          varying vec3 vColor;
          varying float vAlpha;
          void main() {
            vColor = color;
            vAlpha = alpha;
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = 4.0 * (10.0 / -mvPosition.z);
            gl_Position = projectionMatrix * mvPosition;
          }
        `}
        fragmentShader={`
          varying vec3 vColor;
          varying float vAlpha;
          void main() {
            vec2 xy = gl_PointCoord.xy - vec2(0.5);
            float ll = length(xy);
            if(ll > 0.5) discard;
            float strength = (0.5 - ll) * 2.0;
            gl_FragColor = vec4(vColor, vAlpha * strength);
          }
        `}
        vertexColors
      />
    </points>
  );
}

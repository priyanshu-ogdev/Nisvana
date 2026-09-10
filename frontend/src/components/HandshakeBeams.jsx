import { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useConnectionStore, CONNECTION_STATES } from '../state/useConnectionStore';

function BeamLine({ start, end, control, isSecure, isHandshaking }) {
   const curve = useMemo(() => new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(...start),
      new THREE.Vector3(...control),
      new THREE.Vector3(...end)
   ), [start, end, control]);
   
   const points = useMemo(() => curve.getPoints(50), [curve]);
   const lineGeometry = useMemo(() => new THREE.BufferGeometry().setFromPoints(points), [points]);
   
   const pulseRef = useRef();
   const pulseCount = 8;
   const dummy = useMemo(() => new THREE.Object3D(), []);
   const offsets = useMemo(() => new Float32Array(pulseCount).map(() => Math.random()), []);
   
   useFrame((state, delta) => {
       if (isSecure && pulseRef.current) {
           for(let i=0; i<pulseCount; i++) {
               offsets[i] += delta * 0.3; 
               if (offsets[i] > 1) offsets[i] -= 1;
               
               const t = offsets[i];
               const pos = curve.getPointAt(t);
               
               dummy.position.copy(pos);
               const scale = Math.sin(t * Math.PI) * 1.5;
               dummy.scale.set(scale, scale, scale);
               dummy.updateMatrix();
               pulseRef.current.setMatrixAt(i, dummy.matrix);
           }
           pulseRef.current.instanceMatrix.needsUpdate = true;
       }
   });

   const teal = new THREE.Color('#00AFA5'); 
   const slate = new THREE.Color('#52657D'); 

   const lineRef = useRef();
   
   useEffect(() => {
      if (lineRef.current) {
         lineRef.current.computeLineDistances();
      }
   }, [lineGeometry]);

   if (!isSecure && !isHandshaking) {
      return (
         <line ref={lineRef} geometry={lineGeometry}>
            <lineDashedMaterial color={slate} dashSize={0.2} gapSize={0.2} opacity={0.12} transparent />
         </line>
      );
   }

   return (
      <group>
         <line geometry={lineGeometry}>
            <lineBasicMaterial color={teal} opacity={0.4} transparent />
         </line>
         {isSecure && (
             <instancedMesh ref={pulseRef} args={[null, null, pulseCount]}>
                <circleGeometry args={[0.04, 8]} />
                <meshBasicMaterial color="#43D7E5" transparent opacity={0.8} />
             </instancedMesh>
         )}
      </group>
   );
}

export function HandshakeBeams() {
   const { clients } = useConnectionStore();
   
   const p1State = clients['person-1']?.state;
   const p2State = clients['person-2']?.state;

   return (
       <group>
          <BeamLine 
            start={[0,0,0]} 
            end={[-4.5, 2.0, 0]} 
            control={[-2, 1, 0]} 
            isSecure={p1State === CONNECTION_STATES.SECURE} 
            isHandshaking={p1State === CONNECTION_STATES.HANDSHAKING}
          />
          <BeamLine 
            start={[0,0,0]} 
            end={[4.5, 2.0, 0]} 
            control={[2, 1, 0]} 
            isSecure={p2State === CONNECTION_STATES.SECURE} 
            isHandshaking={p2State === CONNECTION_STATES.HANDSHAKING}
          />
       </group>
   );
}

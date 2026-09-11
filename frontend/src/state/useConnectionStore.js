import { create } from 'zustand';

export const CONNECTION_STATES = {
  DORMANT: 'DORMANT',
  HANDSHAKING: 'HANDSHAKING',
  SECURE: 'SECURE',
  DROPPED: 'DROPPED',
};

export const GLOBAL_STATES = {
  OFFLINE: 'OFFLINE',
  NO_LINK: 'NO_LINK',
  PARTIAL_LINK: 'PARTIAL_LINK',
  STREAMING: 'STREAMING'
};

export const fftStreams = {};

export const useConnectionStore = create((set, get) => ({
  ws: null,
  isSimulated: false,
  globalState: GLOBAL_STATES.OFFLINE,
  rtt_ms: null,
  _pingInterval: null,

  telemetry: {
    latency_ms: null,
    inference_ms: null,
    network_ms: null,
    node_rtt_ms: null,
    dropped_frames: 0,
    queue_depth: 0,
    link_quality: 'healthy',
    snr_improvement_db: null,
    model: null,
    pi_cpu_temp: null,
    cpu_pct: null,
    ram_pct: null,
    backend: null,
    provider: null,
    algorithmic_delay_ms: null,
    real_time_factor: null,
    thermal_tier: null,
    degradation_reason: null,
    aec_mode: null,
  },
  
  // Clients are now dynamically populated from node_online events
  clients: {},

  initConnection: () => {
    const { ws, _startSimulation } = get();
    if (ws) ws.close();

    // Demo mode is explicit so an unavailable model/backend never looks live.
    if (import.meta.env.VITE_DEMO_MODE === '1') {
      _startSimulation();
      return;
    }

    const wsUrl = import.meta.env.VITE_BACKEND_WS_URL || import.meta.env.VITE_HUB_WS_URL || 'ws://127.0.0.1:8001/dashboard';
    
    if (wsUrl) {
      try {
        const socket = new WebSocket(wsUrl);
        socket.binaryType = 'arraybuffer';
        
        let connectTimeout = setTimeout(() => {
          if (socket.readyState !== WebSocket.OPEN) {
             socket.close();
             _startSimulation();
          }
        }, 3000);

        socket.onopen = () => {
          clearTimeout(connectTimeout);
          set({ ws: socket, isSimulated: false, globalState: GLOBAL_STATES.NO_LINK, clients: {} });
          get().sendPing();
        };

        socket.onmessage = async (event) => {
          if (event.data instanceof ArrayBuffer) {
             get()._processBinaryMessage(event.data);
          } else {
             try {
               const msg = JSON.parse(event.data);
               get()._processMessage(msg);
             } catch (e) {
                console.error("Invalid WS JSON message", e);
             }
          }
        };

        socket.onerror = () => {
          clearTimeout(connectTimeout);
          _startSimulation();
        };

        socket.onclose = () => {
          clearTimeout(connectTimeout);
          const pingInterval = get()._pingInterval;
          if (pingInterval) clearInterval(pingInterval);
          set({ globalState: GLOBAL_STATES.OFFLINE, ws: null, rtt_ms: null, _pingInterval: null });
          
          setTimeout(() => {
             get().initConnection();
          }, 5000);
        };
      } catch (e) {
        _startSimulation();
      }
    } else {
      _startSimulation();
    }
  },

  reconnect: () => {
    const { ws } = get();
    if (ws) ws.close();
    get().initConnection();
  },

  sendPing: () => {
    const interval = setInterval(() => {
      const { ws } = get();
      if (ws && ws.readyState === WebSocket.OPEN) {
        const seq = Date.now();
        ws.send(JSON.stringify({ type: 'ping', seq, timestamp: Date.now() }));
      }
    }, 1000);
    set({ _pingInterval: interval });
  },

  _processBinaryMessage: (buffer) => {
      // Wire framing format:
      // Byte 0: frame_type (1 = FFT Stream, 2 = Raw Audio Passthrough, 3 = HW Health)
      // Byte 1: node_id length (L)
      // Bytes 2..2+L-1: node_id (utf-8)
      // Bytes 2+L..2+L+7: timestamp_ms (uint64, little endian)
      // Remaining bytes: Frame payload
      if (!buffer || buffer.byteLength < 2) return;
      const view = new DataView(buffer);
      const frame_type = view.getUint8(0);
      const id_len = view.getUint8(1);
      const header_len = 2 + id_len + 8;
      if (buffer.byteLength < header_len) return;

      const decoder = new TextDecoder('utf-8');
      const id_bytes = new Uint8Array(buffer, 2, id_len);
      const node_id = decoder.decode(id_bytes);
      const timestamp_ms = Number(view.getBigUint64(2 + id_len, true));

      if (frame_type === 1) {
          // FRAME_TYPE_FFT: 64B raw + 64B enhanced
          if (buffer.byteLength < header_len + 128) return;
          if (!fftStreams[node_id]) {
              fftStreams[node_id] = { enhanced: new Uint8Array(64), raw: new Uint8Array(64) };
          }
          const raw = new Uint8Array(buffer, header_len, 64);
          const enhanced = new Uint8Array(buffer, header_len + 64, 64);
          for (let i = 0; i < 64; i++) {
              fftStreams[node_id].raw[i] = raw[i];
              fftStreams[node_id].enhanced[i] = enhanced[i];
          }
      } else if (frame_type === 2) {
          // FRAME_TYPE_AUDIO: raw 16-bit PCM audio passthrough
          const audioData = buffer.slice(header_len);
          if (typeof window !== 'undefined') {
              window.dispatchEvent(new CustomEvent('aegis-audio-frame', {
                  detail: { nodeId: node_id, timestamp: timestamp_ms, audio: audioData }
              }));
          }
      } else if (frame_type === 3) {
          // FRAME_TYPE_HEALTH: high-rate HW health telemetry
          const healthData = buffer.slice(header_len);
          if (typeof window !== 'undefined') {
              window.dispatchEvent(new CustomEvent('aegis-health-frame', {
                  detail: { nodeId: node_id, timestamp: timestamp_ms, payload: healthData }
              }));
          }
      }
  },

  _processMessage: (msg) => {
    const { clients } = get();
    
    if (msg.type === 'node_online') {
        const nodeId = msg.node_id;
        if (!fftStreams[nodeId]) {
            fftStreams[nodeId] = { enhanced: new Uint8Array(64), raw: new Uint8Array(64) };
        }
        set(state => ({
            clients: {
                ...state.clients,
                [nodeId]: { 
                   state: msg.secure ? CONNECTION_STATES.SECURE : CONNECTION_STATES.DORMANT,
                   muted: {}, 
                   hw: msg.hw || {},
                   anc: { active: false, vad: false, out: null, in: null, sidetone: false } 
                }
            }
        }));
        get()._evalGlobalState(msg.secure);
    }
    else if (msg.type === 'node_offline') {
        const nodeId = msg.node_id;
        delete fftStreams[nodeId];
        set(state => {
            const newClients = { ...state.clients };
            delete newClients[nodeId];
            return { clients: newClients };
        });
        get()._evalGlobalState();
    }
    else if (msg.type === 'pong') {
       const pingTs = msg.timestamp || msg.ts;
       const rtt = pingTs ? (Date.now() - pingTs) : (typeof msg.rtt_ms === 'number' ? msg.rtt_ms : null);
       if (typeof rtt === 'number' && !isNaN(rtt)) {
          set({ rtt_ms: Math.max(0, rtt) });
       }
    }
    else if (msg.type === 'subscription_ack') {
       // Confirmation from Hub of targeted dashboard subscriptions
       console.log('Subscriptions confirmed:', msg.nodes);
    }
    else if (msg.type === 'handshake_ack') {
       const clientId = msg.clientId || msg.node_id;
       window._aegisShockwave = { x: 0, y: 0, strength: 15.0 };
       set(state => {
          const newClients = { ...state.clients };
          if(newClients[clientId]) {
             newClients[clientId].state = CONNECTION_STATES.SECURE;
          }
          return { clients: newClients };
       });
       get()._evalGlobalState(true);
    }
    else if (msg.type === 'anc_state') {
       const clientId = msg.clientId || msg.node_id;
       const client = clients[clientId];
       if (client) {
           set(state => {
              const newClients = { ...state.clients };
              newClients[clientId].anc = {
                 active: msg.anc_active,
                 vad: msg.vad_speech,
                 out: msg.ambient_out_db,
                 in: msg.ambient_in_ear_db,
                 sidetone: msg.sidetone_on
              };
              return { clients: newClients };
           });
       }
    }
    else if (msg.type === 'telemetry') {
       const currentRtt = get().rtt_ms;
       const netMs = (typeof msg.network_ms === 'number')
          ? msg.network_ms
          : (currentRtt != null ? currentRtt / 2 : null);
       const nextTelemetry = {
          latency_ms: msg.latency_ms,
          inference_ms: msg.inference_ms ?? null,
          network_ms: netMs,
          node_rtt_ms: msg.node_rtt_ms ?? null,
          dropped_frames: msg.dropped_frames ?? 0,
          queue_depth: msg.queue_depth ?? 0,
          link_quality: msg.link_quality ?? 'healthy',
          snr_improvement_db: msg.snr_improvement_db,
          model: msg.model,
          pi_cpu_temp: msg.pi_cpu_temp ?? null,
          cpu_pct: msg.cpu_pct ?? null,
          ram_pct: msg.ram_pct ?? null,
          backend: msg.backend ?? null,
          provider: msg.provider ?? null,
          algorithmic_delay_ms: msg.algorithmic_delay_ms ?? null,
          real_time_factor: msg.real_time_factor ?? null,
          thermal_tier: msg.thermal_tier ?? null,
          degradation_reason: msg.degradation_reason ?? null,
          aec_mode: msg.aec_mode ?? null,
       };
       set({ telemetry: nextTelemetry });
       const clientId = msg.clientId || msg.node_id;
       if (clientId && get().clients[clientId]) {
          set(state => ({
             clients: {
                ...state.clients,
                [clientId]: {
                   ...state.clients[clientId],
                   telemetry: nextTelemetry,
                },
             },
          }));
       }
    }
    else if (msg.type === 'link_status') {
       if (msg.state === 'streaming') {
           get()._evalGlobalState(true);
       }
    }
  },

  subscribeToNodes: (nodes) => {
    const { ws } = get();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'subscribe', nodes }));
    }
  },

  _evalGlobalState: (isStreaming = false) => {
     const { clients, isSimulated } = get();
     if (isSimulated) return;

     const clientKeys = Object.keys(clients);
     if (clientKeys.length === 0) {
         set({ globalState: GLOBAL_STATES.NO_LINK });
         return;
     }

     const allSecure = clientKeys.every(k => clients[k].state === CONNECTION_STATES.SECURE);
     const anySecure = clientKeys.some(k => clients[k].state === CONNECTION_STATES.SECURE);

     if (allSecure) {
        set({ globalState: isStreaming ? GLOBAL_STATES.STREAMING : GLOBAL_STATES.PARTIAL_LINK });
     } else if (anySecure) {
        set({ globalState: GLOBAL_STATES.PARTIAL_LINK });
     } else {
        set({ globalState: GLOBAL_STATES.NO_LINK });
     }
  },

  sendHandshakeInit: (clientId) => {
     const { ws, isSimulated } = get();
     
     set(state => {
         const newClients = { ...state.clients };
         if (newClients[clientId]) {
            newClients[clientId].state = CONNECTION_STATES.HANDSHAKING;
         }
         return { clients: newClients };
     });

     if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'handshake_init', clientId, timestamp: Date.now() }));
     } else if (isSimulated) {
        setTimeout(() => {
           get()._processMessage({ type: 'handshake_ack', clientId, status: 'ok' });
           const clients = get().clients;
           const allSecure = Object.values(clients).every(c => c.state === CONNECTION_STATES.SECURE);
           if (allSecure) {
               get()._processMessage({ type: 'link_status', state: 'streaming' });
           }
        }, 1200);
     }
  },

  sendHardwareMute: (clientId, target, muteState) => {
     const { ws } = get();
     set(store => {
         const newClients = { ...store.clients };
         if (newClients[clientId]) {
             const prevMuted = newClients[clientId]?.muted || {};
             newClients[clientId] = {
               ...newClients[clientId],
               muted: { ...prevMuted, [target]: muteState },
             };
         }
         return { clients: newClients };
     });

     if (ws && ws.readyState === WebSocket.OPEN) {
        // Targeted control message relayed by Hub
        ws.send(JSON.stringify({ type: 'hardware_mute', node_id: clientId, clientId, target, state: muteState }));
     }
  },

  sendAncSet: (clientId, enabled) => {
     const { ws } = get();
     set(store => {
         const newClients = { ...store.clients };
         if (newClients[clientId]) {
             const prevAnc = newClients[clientId]?.anc || {};
             newClients[clientId] = {
               ...newClients[clientId],
               anc: { ...prevAnc, active: enabled },
             };
         }
         return { clients: newClients };
     });
     if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'anc_set', node_id: clientId, clientId, enabled }));
     }
  },

  _startSimulation: () => {
    fftStreams['person-1'] = { enhanced: new Uint8Array(64), raw: new Uint8Array(64) };
    fftStreams['person-2'] = { enhanced: new Uint8Array(64), raw: new Uint8Array(64) };

    let t = 0;
    const simInterval = setInterval(() => {
       t += 0.05;
       ['person-1', 'person-2'].forEach((id, pIdx) => {
          if (!fftStreams[id]) return;
          const isSec = get().clients[id]?.state === CONNECTION_STATES.SECURE;
          for (let i = 0; i < 64; i++) {
             if (isSec) {
                const wave = Math.sin(t * 2 + i * 0.2 + pIdx) * 0.5 + 0.5;
                const noise = Math.random() * 0.15;
                fftStreams[id].raw[i] = Math.floor((wave * 0.8 + noise) * 200);
                fftStreams[id].enhanced[i] = Math.floor((wave * 0.95 + noise * 0.2) * 240);
             } else {
                fftStreams[id].raw[i] = 0;
                fftStreams[id].enhanced[i] = 0;
             }
          }
       });
    }, 33);

    set({ 
      isSimulated: true, 
      globalState: GLOBAL_STATES.OFFLINE,
      telemetry: {
         latency_ms: 8.4,
         inference_ms: 6.2,
         network_ms: 2.2,
         node_rtt_ms: 3.2,
         dropped_frames: 0,
         queue_depth: 0,
         link_quality: 'healthy',
         snr_improvement_db: null,
         model: 'DeepFilterNet3',
         pi_cpu_temp: null,
         cpu_pct: null,
         ram_pct: null,
         backend: 'simulation',
         provider: 'simulation',
         algorithmic_delay_ms: null,
         real_time_factor: null,
         thermal_tier: null,
         degradation_reason: 'no_live_backend',
         aec_mode: null,
      },
      clients: {
        'person-1': { 
          state: CONNECTION_STATES.DORMANT, 
          muted: {}, 
          hw: { alsainputs: ['Primary Mic (Simulated)'], alsaoutputs: ['Headset DAC (Simulated)'] },
          anc: { active: false, vad: false, out: 34.0, in: 18.0, sidetone: false } 
        },
        'person-2': { 
          state: CONNECTION_STATES.DORMANT, 
          muted: {}, 
          hw: { alsainputs: ['Reference Mic (Simulated)'], alsaoutputs: ['Headset DAC (Simulated)'] },
          anc: { active: false, vad: false, out: 30.0, in: 15.0, sidetone: false } 
        }
      }
    });
  }
}));

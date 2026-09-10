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
    snr_improvement_db: null,
    model: null,
    pi_cpu_temp: null,
    cpu_pct: null,
    ram_pct: null,
  },
  
  // Clients are now dynamically populated from node_online events
  clients: {},

  initConnection: () => {
    const { ws, _startSimulation } = get();
    if (ws) ws.close();

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
      // P1: Binary framing format:
      // <BB + len(node_id_bytes)s + Q + 64B + 64B
      const view = new DataView(buffer);
      const frame_type = view.getUint8(0);
      if (frame_type === 1) {
          const id_len = view.getUint8(1);
          const decoder = new TextDecoder('utf-8');
          const id_bytes = new Uint8Array(buffer, 2, id_len);
          const node_id = decoder.decode(id_bytes);
          
          let offset = 2 + id_len + 8; // skip timestamp for now
          
          if (!fftStreams[node_id]) return;
          
          const raw = new Uint8Array(buffer, offset, 64);
          const enhanced = new Uint8Array(buffer, offset + 64, 64);
          
          for (let i = 0; i < 64; i++) {
              fftStreams[node_id].raw[i] = raw[i];
              fftStreams[node_id].enhanced[i] = enhanced[i];
          }
      }
  },

  _processMessage: (msg) => {
    const { clients } = get();
    
    if (msg.type === 'node_online') {
        const nodeId = msg.node_id;
        fftStreams[nodeId] = { enhanced: new Uint8Array(64), raw: new Uint8Array(64) };
        set(state => ({
            clients: {
                ...state.clients,
                [nodeId]: { 
                   state: CONNECTION_STATES.SECURE, // automatically secure for now 
                   muted: {}, 
                   hw: msg.hw || {},
                   anc: { active: false, vad: false, out: null, in: null, sidetone: false } 
                }
            }
        }));
        get()._evalGlobalState(true); // Treat as streaming if secure
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
       const rtt = Date.now() - msg.timestamp;
       set({ rtt_ms: rtt });
    }
    else if (msg.type === 'handshake_ack') {
       window._aegisShockwave = { x: 0, y: 0, strength: 15.0 };
       set(state => {
          const newClients = { ...state.clients };
          if(newClients[msg.clientId]) {
             newClients[msg.clientId].state = CONNECTION_STATES.SECURE;
          }
          return { clients: newClients };
       });
       get()._evalGlobalState();
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
       set({ telemetry: {
          latency_ms: msg.latency_ms,
          inference_ms: msg.inference_ms,
          network_ms: get().rtt_ms ? (get().rtt_ms / 2) : msg.network_ms,
          snr_improvement_db: msg.snr_improvement_db,
          model: msg.model,
          pi_cpu_temp: msg.pi_cpu_temp ?? null,
          cpu_pct: msg.cpu_pct ?? null,
          ram_pct: msg.ram_pct ?? null,
       }});
    }
    else if (msg.type === 'link_status') {
       if (msg.state === 'streaming') {
           get()._evalGlobalState(true);
       }
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
           get()._processMessage({ type: 'node_online', node_id: clientId, hw: { mic_primary: true } });
           get()._processMessage({ type: 'link_status', state: 'streaming' });
        }, 1500);
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
        ws.send(JSON.stringify({ type: 'hardware_mute', node_id: clientId, target, state: muteState }));
     }
  },

  sendAncSet: (clientId, enabled) => {
     const { ws } = get();
     if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'anc_set', node_id: clientId, enabled }));
     }
  },

  _startSimulation: () => {
    set({ isSimulated: true, globalState: GLOBAL_STATES.OFFLINE });
  }
}));


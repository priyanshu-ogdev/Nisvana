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

export const fftStreams = {
  'person-1': new Uint8Array(64),
  'person-2': new Uint8Array(64)
};

export const useConnectionStore = create((set, get) => ({
  ws: null,
  isSimulated: false,
  globalState: GLOBAL_STATES.OFFLINE,
  
  telemetry: {
    latency_ms: null,
    snr_improvement_db: null,
    model: null,
    pi_cpu_temp: null,
  },
  
  hw_status: {
    headset_detected: true,
    mic_primary: true,
    mic_reference: true,
    mic_throat: true
  },

  clients: {
    'person-1': { state: CONNECTION_STATES.DORMANT, muted: false, anc: { active: false, vad: false, out: null, in: null, sidetone: false } },
    'person-2': { state: CONNECTION_STATES.DORMANT, muted: false, anc: { active: false, vad: false, out: null, in: null, sidetone: false } }
  },

  initConnection: () => {
    const { ws, _startSimulation } = get();
    if (ws) ws.close();

    const wsUrl = import.meta.env.VITE_BACKEND_WS_URL || import.meta.env.VITE_PI_WS_URL;
    
    if (wsUrl) {
      try {
        const socket = new WebSocket(wsUrl);
        let connectTimeout = setTimeout(() => {
          if (socket.readyState !== WebSocket.OPEN) {
             socket.close();
             _startSimulation();
          }
        }, 3000);

        socket.onopen = () => {
          clearTimeout(connectTimeout);
          set({ ws: socket, isSimulated: false, globalState: GLOBAL_STATES.NO_LINK });
        };

        socket.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            get()._processMessage(msg);
          } catch (e) {
             console.error("Invalid WS message", e);
          }
        };

        socket.onerror = () => {
          clearTimeout(connectTimeout);
          _startSimulation();
        };

        socket.onclose = () => {
          clearTimeout(connectTimeout);
          set({ globalState: GLOBAL_STATES.OFFLINE, ws: null });
          
          // Reconnect attempt every 5s
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

  _processMessage: (msg) => {
    const { clients } = get();
    
    if (msg.type === 'handshake_ack') {
       window._aegisShockwave = { x: 0, y: 0, strength: 15.0 };
       set(state => {
          const newClients = { ...state.clients };
          newClients[msg.clientId] = { ...newClients[msg.clientId], state: CONNECTION_STATES.SECURE };
          return { clients: newClients };
       });
       get()._evalGlobalState();
    }
    else if (msg.type === 'hw_status') {
       set({ hw_status: {
          headset_detected: msg.headset_detected,
          mic_primary: msg.mic_primary,
          mic_reference: msg.mic_reference,
          mic_throat: msg.mic_throat
       }});
    }
    else if (msg.type === 'fft_stream') {
       const client = clients[msg.clientId];
       if (client && client.state === CONNECTION_STATES.SECURE) {
           const bins = msg.bins;
           for (let i = 0; i < 64; i++) {
               fftStreams[msg.clientId][i] = bins[i] || 0;
           }
       } else {
           fftStreams[msg.clientId].fill(0);
       }
    }
    else if (msg.type === 'anc_state') {
       const client = clients[msg.clientId];
       if (client && client.state === CONNECTION_STATES.SECURE) {
           set(state => {
              const newClients = { ...state.clients };
              newClients[msg.clientId].anc = {
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
          snr_improvement_db: msg.snr_improvement_db,
          model: msg.model,
          pi_cpu_temp: msg.pi_cpu_temp
       }});
    }
    else if (msg.type === 'link_status') {
       if (msg.state === 'streaming') {
           get()._evalGlobalState(true);
       }
    }
  },

  _evalGlobalState: (isStreaming = false) => {
     const { clients, isSimulated, globalState } = get();
     if (isSimulated) {
        return; 
     }

     const p1Secure = clients['person-1'].state === CONNECTION_STATES.SECURE;
     const p2Secure = clients['person-2'].state === CONNECTION_STATES.SECURE;

     if (p1Secure && p2Secure) {
        set({ globalState: isStreaming ? GLOBAL_STATES.STREAMING : GLOBAL_STATES.PARTIAL_LINK });
     } else if (p1Secure || p2Secure) {
        set({ globalState: GLOBAL_STATES.PARTIAL_LINK });
     } else {
        set({ globalState: GLOBAL_STATES.NO_LINK });
     }
  },

  sendHandshakeInit: (clientId) => {
     const { ws, isSimulated } = get();
     
     set(state => {
         const newClients = { ...state.clients };
         newClients[clientId] = { ...newClients[clientId], state: CONNECTION_STATES.HANDSHAKING };
         return { clients: newClients };
     });

     if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'handshake_init', clientId, timestamp: Date.now() }));
     } else if (isSimulated) {
        setTimeout(() => {
           get()._processMessage({ type: 'handshake_ack', clientId, status: 'ok' });
           if (get().clients['person-1'].state === CONNECTION_STATES.SECURE && get().clients['person-2'].state === CONNECTION_STATES.SECURE) {
               get()._processMessage({ type: 'link_status', state: 'streaming' });
           }
        }, 1500);
     }
  },

  sendHardwareMute: (clientId, state) => {
     const { ws } = get();
     set(store => {
         const newClients = { ...store.clients };
         newClients[clientId] = { ...newClients[clientId], muted: state };
         return { clients: newClients };
     });

     if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'hardware_mute', clientId, state }));
     }
  },

  sendAncSet: (clientId, enabled) => {
     const { ws } = get();
     if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'anc_set', clientId, enabled }));
     }
  },

  _startSimulation: () => {
    set({ isSimulated: true, globalState: GLOBAL_STATES.OFFLINE });
  }
}));

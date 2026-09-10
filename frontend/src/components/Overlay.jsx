import { useConnectionStore, GLOBAL_STATES, CONNECTION_STATES } from '../state/useConnectionStore';
import { ShieldCheck, Zap, Radio, AlertTriangle, RefreshCw } from 'lucide-react';
import { PersonCard } from './PersonCard';

export function Overlay() {
  const { globalState, isSimulated, telemetry, clients, sendHandshakeInit, rtt_ms, reconnect } = useConnectionStore();

  const getGlobalStatusDisplay = () => {
    if (isSimulated || globalState === GLOBAL_STATES.OFFLINE) {
        return { text: 'OFFLINE / SIMULATED DEMO', color: 'text-[var(--amber)]', dot: 'bg-[var(--amber)] animate-pulse', icon: <AlertTriangle size={14} className="mr-2" /> };
    }
    switch (globalState) {
        case GLOBAL_STATES.NO_LINK: return { text: 'HUB REACHED — NO NODES', color: 'text-[var(--text-low)]', dot: 'bg-[var(--text-low)]', icon: <Radio size={14} className="mr-2" /> };
        case GLOBAL_STATES.PARTIAL_LINK: return { text: 'NODES PENDING HANDSHAKE', color: 'text-[var(--amber)]', dot: 'bg-[var(--amber)] animate-pulse', icon: <Zap size={14} className="mr-2" /> };
        case GLOBAL_STATES.STREAMING: return { text: 'STREAMING', color: 'text-[var(--mint)]', dot: 'bg-[var(--mint)] shadow-[0_0_8px_var(--mint)]', icon: <ShieldCheck size={14} className="mr-2" /> };
        default: return { text: 'UNKNOWN', color: 'text-[var(--text-low)]', dot: 'bg-[var(--text-low)]', icon: <Radio size={14} className="mr-2" /> };
    }
  };

  const statusDisplay = getGlobalStatusDisplay();

  const clientKeys = Object.keys(clients);

  const handleInitHandshake = () => {
     clientKeys.forEach(clientId => {
         if (clients[clientId]?.state === CONNECTION_STATES.DORMANT) {
             sendHandshakeInit(clientId);
         }
     });
  };
  
  const allSecure = clientKeys.length > 0 && clientKeys.every(k => clients[k].state === CONNECTION_STATES.SECURE);
  const isOffline = isSimulated || globalState === GLOBAL_STATES.OFFLINE;

  return (
    <div className="absolute inset-0 pointer-events-none flex flex-col justify-between overflow-hidden">
      {/* TOP BAR */}
      <div className="h-[56px] px-8 flex justify-between items-center bg-[#0B0E17]/40 backdrop-blur-md border-b border-white/5">
        <div className="flex items-center space-x-3">
          <div className="w-2 h-2 rounded-full bg-[var(--cyan)] shadow-[0_0_8px_var(--cyan)]" />
          <h1 className="font-heading font-semibold text-lg tracking-wider text-[var(--text-hi)]">PROJECT AEGIS</h1>
        </div>
        
        <div className="flex items-center space-x-4">
           {isOffline && (
             <button
               onClick={() => { reconnect?.(); }}
               className="pointer-events-auto flex items-center space-x-1.5 px-3 py-1.5 rounded text-xs font-bold tracking-widest uppercase border bg-[var(--amber)]/10 border-[var(--amber)]/30 text-[var(--amber)] hover:bg-[var(--amber)]/20 transition-all"
             >
               <RefreshCw size={11} />
               <span>RECONNECT HUB</span>
             </button>
           )}

           <button 
             onClick={handleInitHandshake}
             disabled={allSecure || clientKeys.length === 0}
             className={`pointer-events-auto px-4 py-1.5 rounded text-xs font-bold tracking-widest uppercase transition-all border
                ${(allSecure || clientKeys.length === 0) ? 'bg-white/5 border-white/5 text-[var(--text-low)]' : 'bg-[var(--cyan)]/10 border-[var(--cyan)]/30 text-[var(--cyan)] hover:bg-[var(--cyan)]/20'}
             `}
           >
              INITIATE NODE HANDSHAKE
           </button>
           
           <div className={`flex items-center text-xs font-bold tracking-widest ${statusDisplay.color}`}>
              {statusDisplay.icon}
              {statusDisplay.text}
              <div className={`ml-3 w-2 h-2 rounded-full ${statusDisplay.dot}`} />
           </div>
        </div>
      </div>

      {/* MID SECTION (CARDS) */}
      <div className="flex-1 relative w-full h-full flex justify-center items-center">
         {clientKeys.length === 0 && !isSimulated && (
             <div className="text-[var(--text-low)] font-mono text-sm tracking-widest">WAITING FOR NODES TO CONNECT...</div>
         )}
         {clientKeys.map((clientId, idx) => (
             <PersonCard 
                key={clientId} 
                clientId={clientId} 
                index={idx}
                totalCount={clientKeys.length}
             />
         ))}
      </div>

      {/* TELEMETRY BAR */}
      <div className="w-full flex justify-center mb-[32px]">
         <div className="h-[64px] bg-[var(--panel)] backdrop-blur-xl border border-[var(--panel-border)] rounded-2xl flex items-center px-8 space-x-6 shadow-[0_4px_30px_rgba(0,0,0,0.5)]">
             <TelemetryCell label="RTT" value={rtt_ms != null ? `${rtt_ms.toFixed(0)} ms` : '—'} color={rtt_ms != null ? "text-[var(--mint)]" : "text-[var(--text-low)]"} />
             <div className="w-px h-8 bg-white/10" />
             <TelemetryCell label="NETWORK" value={telemetry.network_ms != null ? `${telemetry.network_ms.toFixed(1)} ms` : '—'} color="text-[var(--cyan)]" />
             <div className="w-px h-8 bg-white/10" />
             <TelemetryCell label="INFERENCE" value={telemetry.inference_ms != null ? `${telemetry.inference_ms.toFixed(1)} ms` : '—'} color="text-[var(--cyan)]" />
             <div className="w-px h-8 bg-white/10" />
             <TelemetryCell label="SNR ▲" value={telemetry.snr_improvement_db != null ? `+${telemetry.snr_improvement_db.toFixed(1)} dB` : '—'} color="text-[var(--mint)]" />
             <div className="w-px h-8 bg-white/10" />
             <TelemetryCell label="PI TEMP" value={telemetry.pi_cpu_temp != null ? `${telemetry.pi_cpu_temp.toFixed(1)} °C` : '—'} />
             <div className="w-px h-8 bg-white/10" />
             <TelemetryCell label="CPU" value={telemetry.cpu_pct != null ? `${telemetry.cpu_pct.toFixed(0)} %` : '—'} />
             <div className="w-px h-8 bg-white/10" />
             <TelemetryCell label="RAM" value={telemetry.ram_pct != null ? `${telemetry.ram_pct.toFixed(0)} %` : '—'} />
             <div className="w-px h-8 bg-white/10" />
             <TelemetryCell label="MODEL" value={telemetry.model || '—'} color="text-[var(--cyan)]" />
         </div>
      </div>
    </div>
  );
}

function TelemetryCell({ label, value, color = "text-[var(--text-hi)]" }) {
  return (
    <div className="flex flex-col items-center min-w-[72px]">
      <div className="text-[9px] text-[var(--text-low)] font-bold tracking-[0.2em] mb-1">{label}</div>
      <div className={`font-mono text-xs ${color}`}>{value}</div>
    </div>
  );
}

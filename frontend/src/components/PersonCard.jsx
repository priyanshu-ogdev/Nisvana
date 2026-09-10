import { useConnectionStore, CONNECTION_STATES } from '../state/useConnectionStore';
import { Headphones, MicOff, Mic, Shield } from 'lucide-react';
import { Waveform2D } from './Waveform2D';

export function PersonCard({ clientId, side }) {
  const { clients, sendHardwareMute, sendAncSet, isSimulated } = useConnectionStore();
  const client = clients[clientId];
  
  if (!client) return null;

  const isSecure = client.state === CONNECTION_STATES.SECURE;
  const isDormant = client.state === CONNECTION_STATES.DORMANT;
  const isHandshaking = client.state === CONNECTION_STATES.HANDSHAKING;

  let statusText, statusColor, dotState;
  
  if (isDormant) {
      statusText = clientId === 'person-2' ? "REMOTE — AWAITING HANDSHAKE" : "AWAITING HANDSHAKE";
      statusColor = "text-[var(--text-low)]";
      dotState = "bg-[var(--text-low)] animate-slow-pulse";
  } else if (isHandshaking) {
      statusText = "HANDSHAKING";
      statusColor = "text-[var(--amber)]";
      dotState = "bg-[var(--amber)] animate-slow-pulse";
  } else {
      statusText = "LINK SECURE";
      statusColor = "text-[var(--mint)]";
      dotState = "bg-[var(--mint)] shadow-[0_0_8px_var(--mint)]";
  }

  if (isSimulated && !isSecure) statusText += " (SIM)";

  const personNum = clientId.split('-')[1];

  const borderClass = isSecure ? 'border-[var(--panel-border-active)] shadow-[0_4px_30px_rgba(0,0,0,0.5),0_0_15px_rgba(95,242,214,0.1)]' : 'border-[var(--panel-border)] shadow-[0_4px_30px_rgba(0,0,0,0.5)]';

  return (
      <div 
        className={`absolute top-[40px] w-[240px] pointer-events-auto backdrop-blur-xl bg-[var(--panel)] border ${borderClass} rounded-[14px] p-4 flex flex-col transition-all duration-300`}
        style={{ [side === 'left' ? 'left' : 'right']: '40px' }}
      >
          {/* Header Row */}
          <div className="flex justify-between items-center mb-3">
             <div className="flex items-center space-x-2">
                <Headphones size={32} className="text-[var(--text-low)]" />
                <div className="text-[var(--text-hi)] font-bold tracking-widest text-xs uppercase">PERSON {personNum}</div>
             </div>
             <div className="flex items-center space-x-1">
                <button 
                  onClick={() => isSecure && sendHardwareMute(clientId, !client.muted)}
                  disabled={!isSecure}
                  className={`p-1.5 rounded transition-colors ${client.muted ? 'bg-[var(--rose)]/20 text-[var(--rose)]' : 'bg-white/5 text-[var(--text-low)] hover:bg-white/10'} ${!isSecure ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  {client.muted ? <MicOff size={14} /> : <Mic size={14} />}
                </button>
                <button 
                  onClick={() => isSecure && sendAncSet(clientId, !client.anc.active)}
                  disabled={!isSecure}
                  className={`p-1.5 rounded transition-colors ${client.anc.active ? 'bg-[var(--cyan)]/20 text-[var(--cyan)]' : 'bg-white/5 text-[var(--text-low)] hover:bg-white/10'} ${!isSecure ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <Shield size={14} />
                </button>
             </div>
          </div>

          {/* Status Line */}
          <div className={`flex items-center space-x-2 text-[10px] font-bold tracking-widest mb-3 ${statusColor}`}>
              <div className={`w-1.5 h-1.5 rounded-full ${dotState}`} />
              <span>{statusText}</span>
          </div>

          {/* Ambience Row */}
          <div className="bg-black/20 rounded-md p-2 mb-3 relative overflow-hidden min-h-[48px] flex flex-col justify-center border border-white/5">
              {/* Meters */}
              <div className="flex justify-between items-center mb-1.5 relative z-10">
                 <div className="flex items-center space-x-2">
                    <span className="text-[8px] text-[var(--text-low)] tracking-widest w-8">OUT</span>
                    {/* Tick track */}
                    <div className="w-[40px] h-1.5 bg-[var(--text-low)]/20 rounded-full overflow-hidden relative border-x border-[var(--text-low)]/50">
                        {isSecure && <div className="h-full bg-[var(--text-mid)] transition-all duration-75" style={{ width: `${Math.min(100, (client.anc.out || 0))}%` }} />}
                    </div>
                 </div>
                 
                 {isSecure && client.anc.active && (
                     <div className="px-1.5 py-0.5 rounded text-[8px] font-bold bg-[var(--cyan)]/20 text-[var(--cyan)]">
                        ANC −{((client.anc.out || 0) - (client.anc.in || 0)).toFixed(0)} dB
                     </div>
                 )}
                 {isSecure && !client.anc.active && (
                     <div className="px-1.5 py-0.5 rounded text-[8px] font-bold bg-white/5 text-[var(--text-low)]">
                        ANC OFF
                     </div>
                 )}
                 {!isSecure && (
                     <span className="text-[8px] font-mono text-[var(--text-low)]">—</span>
                 )}
              </div>
              
              <div className="flex justify-between items-center relative z-10">
                 <div className="flex items-center space-x-2">
                     <span className="text-[8px] text-[var(--text-low)] tracking-widest w-8">IN-EAR</span>
                     {/* Tick track */}
                     <div className="w-[40px] h-1.5 bg-[var(--text-low)]/20 rounded-full overflow-hidden relative border-x border-[var(--text-low)]/50">
                         {isSecure && <div className="h-full bg-[var(--mint)] transition-all duration-75" style={{ width: `${Math.min(100, (client.anc.in || 0))}%` }} />}
                     </div>
                 </div>
                 {!isSecure && (
                     <span className="text-[8px] font-mono text-[var(--text-low)]">—</span>
                 )}
              </div>

              {/* VAD Speech overlay */}
              {isSecure && client.anc.active && client.anc.vad && (
                  <div className="absolute inset-0 bg-[var(--cyan)]/15 backdrop-blur-[1px] z-20 flex items-center justify-center animate-pulse">
                     <span className="text-[8px] font-bold text-[var(--cyan)] tracking-widest">WORLD SILENCED {client.anc.sidetone ? '· SIDETONE ON' : ''}</span>
                  </div>
              )}
          </div>

          {/* Waveform Docked Container */}
          <div className="h-[56px] w-full relative bg-black/30 rounded border border-white/5 overflow-hidden">
              {isSecure ? (
                  <Waveform2D clientId={clientId} isMuted={client.muted} isSecure={isSecure} />
              ) : (
                  <>
                      {/* Dormant / Idle Designed State */}
                      {/* Grid / baseline */}
                      <div className="absolute inset-x-0 bottom-0 h-px bg-[var(--text-low)]/40 shadow-[0_-12px_0_0_rgba(82,101,125,0.2)]" />
                      <div className="absolute inset-x-0 bottom-0 h-[24px] bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNCIgaGVpZ2h0PSI0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxyZWN0IHdpZHRoPSIxIiBoZWlnaHQ9IjEiIGZpbGw9InJnYmEoMjU1LDI1NSwyNTUsMC4wNSkiLz48L3N2Zz4=')] opacity-50" />
                      
                      {/* CSS Scanline */}
                      <div className="absolute top-0 bottom-0 w-[40px] bg-gradient-to-r from-transparent via-[var(--teal)]/30 to-transparent animate-sweep pointer-events-none" />
                      
                      {/* Label */}
                      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                          <span className="text-[8px] text-[var(--text-low)] tracking-widest uppercase font-bold">NO LINK — SIGNAL GATED</span>
                      </div>
                  </>
              )}

              {isSecure && client.muted && (
                   <div className="absolute inset-0 flex items-center justify-center text-[var(--rose)] text-[10px] font-bold tracking-widest opacity-80 pointer-events-none z-30">
                     MUTED
                   </div>
              )}
          </div>
      </div>
  );
}

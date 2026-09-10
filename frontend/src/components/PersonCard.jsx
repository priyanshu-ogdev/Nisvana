import { useRef, useState } from 'react';
import { useConnectionStore, CONNECTION_STATES } from '../state/useConnectionStore';
import { Headphones, MicOff, Mic, Shield, ChevronDown } from 'lucide-react';
import { Waveform2D } from './Waveform2D';

const MUTE_TARGETS = [
  { key: 'primary_mic',     label: 'Primary Mic' },
  { key: 'reference_mic',   label: 'Reference Mic' },
  { key: 'throat_mic',      label: 'Throat Mic' },
  { key: 'headset_output',  label: 'Headset Output' },
];

export function PersonCard({ clientId, side }) {
  const { clients, sendHardwareMute, sendAncSet, isSimulated, hw_status } = useConnectionStore();
  const client = clients[clientId];
  
  // F-1: Long-press mute dropdown state
  const [muteDropdownOpen, setMuteDropdownOpen] = useState(false);
  const longPressTimer = useRef(null);

  // F-7: ALSA tooltip state
  const [alsaTooltipVisible, setAlsaTooltipVisible] = useState(false);
  
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
  const borderClass = isSecure
    ? 'border-[var(--panel-border-active)] shadow-[0_4px_30px_rgba(0,0,0,0.5),0_0_15px_rgba(95,242,214,0.1)]'
    : 'border-[var(--panel-border)] shadow-[0_4px_30px_rgba(0,0,0,0.5)]';

  // Whether any mute target is active
  const anyMuted = Object.values(client.muted || {}).some(Boolean);

  // F-1: Long-press handlers
  const handleMutePointerDown = () => {
    longPressTimer.current = setTimeout(() => {
      setMuteDropdownOpen(true);
    }, 500);
  };
  const handleMutePointerUp = () => {
    clearTimeout(longPressTimer.current);
    if (!muteDropdownOpen) {
      // Short press = toggle primary_mic
      if (isSecure) sendHardwareMute(clientId, 'primary_mic', !client.muted?.primary_mic);
    }
  };
  const handleMuteTarget = (targetKey) => {
    const current = client.muted?.[targetKey] ?? false;
    sendHardwareMute(clientId, targetKey, !current);
    setMuteDropdownOpen(false);
  };

  // ALSA tooltip content
  const alsaInputs = hw_status?.alsainputs || [];
  const alsaOutputs = hw_status?.alsaoutputs || [];

  return (
      <div 
        className={`absolute top-[40px] w-[260px] pointer-events-auto backdrop-blur-xl bg-[var(--panel)] border ${borderClass} rounded-[14px] p-4 flex flex-col transition-all duration-300`}
        style={{ [side === 'left' ? 'left' : 'right']: '40px' }}
        onClick={() => muteDropdownOpen && setMuteDropdownOpen(false)}
      >
          {/* Header Row */}
          <div className="flex justify-between items-center mb-3">
             <div className="flex items-center space-x-2">
                {/* F-7: Headphone icon shows ALSA tooltip on hover */}
                <div 
                  className="relative"
                  onMouseEnter={() => setAlsaTooltipVisible(true)}
                  onMouseLeave={() => setAlsaTooltipVisible(false)}
                >
                  <Headphones size={32} className="text-[var(--text-low)] cursor-help" />
                  {alsaTooltipVisible && (alsaInputs.length > 0 || alsaOutputs.length > 0) && (
                    <div className="absolute left-0 top-full mt-2 z-50 bg-[#0B0E17] border border-[var(--cyan)]/20 rounded-lg p-3 text-[9px] font-mono text-[var(--text-low)] w-[180px] shadow-xl">
                      <div className="text-[var(--cyan)] font-bold mb-1 tracking-widest">ALSA INPUTS</div>
                      {alsaInputs.length > 0 ? alsaInputs.map((n, i) => (
                        <div key={i} className="truncate mb-0.5">{n}</div>
                      )) : <div>none detected</div>}
                      <div className="text-[var(--cyan)] font-bold mt-2 mb-1 tracking-widest">ALSA OUTPUTS</div>
                      {alsaOutputs.length > 0 ? alsaOutputs.map((n, i) => (
                        <div key={i} className="truncate mb-0.5">{n}</div>
                      )) : <div>none detected</div>}
                    </div>
                  )}
                </div>
                <div className="text-[var(--text-hi)] font-bold tracking-widest text-xs uppercase">PERSON {personNum}</div>
             </div>
             <div className="flex items-center space-x-1">
                {/* F-1: Mute button with long-press dropdown */}
                <div className="relative">
                  <button 
                    onPointerDown={handleMutePointerDown}
                    onPointerUp={handleMutePointerUp}
                    onPointerLeave={() => clearTimeout(longPressTimer.current)}
                    disabled={!isSecure}
                    className={`p-1.5 rounded transition-colors flex items-center space-x-0.5
                      ${anyMuted ? 'bg-[var(--rose)]/20 text-[var(--rose)]' : 'bg-white/5 text-[var(--text-low)] hover:bg-white/10'}
                      ${!isSecure ? 'opacity-50 cursor-not-allowed' : ''}`}
                    title="Click to mute primary mic. Long-press for more options."
                  >
                    {anyMuted ? <MicOff size={14} /> : <Mic size={14} />}
                    <ChevronDown size={9} className="opacity-50" />
                  </button>

                  {/* Long-press dropdown */}
                  {muteDropdownOpen && (
                    <div className="absolute right-0 top-full mt-1 z-50 bg-[#0B0E17] border border-white/10 rounded-lg overflow-hidden shadow-xl min-w-[140px]">
                      {MUTE_TARGETS.map(({ key, label }) => {
                        const isMuted = client.muted?.[key] ?? false;
                        return (
                          <button
                            key={key}
                            onClick={() => handleMuteTarget(key)}
                            className={`w-full text-left px-3 py-2 text-[10px] font-mono flex items-center justify-between
                              hover:bg-white/5 transition-colors
                              ${isMuted ? 'text-[var(--rose)]' : 'text-[var(--text-low)]'}`}
                          >
                            <span>{label}</span>
                            {isMuted && <MicOff size={10} />}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>

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
                     <div className="px-1.5 py-0.5 rounded text-[8px] font-bold bg-white/5 text-[var(--text-low)]">ANC OFF</div>
                 )}
                 {!isSecure && <span className="text-[8px] font-mono text-[var(--text-low)]">—</span>}
              </div>
              
              <div className="flex justify-between items-center relative z-10">
                 <div className="flex items-center space-x-2">
                     <span className="text-[8px] text-[var(--text-low)] tracking-widest w-8">IN-EAR</span>
                     <div className="w-[40px] h-1.5 bg-[var(--text-low)]/20 rounded-full overflow-hidden relative border-x border-[var(--text-low)]/50">
                         {isSecure && <div className="h-full bg-[var(--mint)] transition-all duration-75" style={{ width: `${Math.min(100, (client.anc.in || 0))}%` }} />}
                     </div>
                 </div>
                 {!isSecure && <span className="text-[8px] font-mono text-[var(--text-low)]">—</span>}
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
                  <Waveform2D clientId={clientId} isMuted={anyMuted} isSecure={isSecure} />
              ) : (
                  <>
                      <div className="absolute inset-x-0 bottom-0 h-px bg-[var(--text-low)]/40 shadow-[0_-12px_0_0_rgba(82,101,125,0.2)]" />
                      <div className="absolute top-0 bottom-0 w-[40px] bg-gradient-to-r from-transparent via-[var(--teal)]/30 to-transparent animate-sweep pointer-events-none" />
                      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                          <span className="text-[8px] text-[var(--text-low)] tracking-widest uppercase font-bold">NO LINK — SIGNAL GATED</span>
                      </div>
                  </>
              )}

              {isSecure && anyMuted && (
                   <div className="absolute inset-0 flex items-center justify-center text-[var(--rose)] text-[10px] font-bold tracking-widest opacity-80 pointer-events-none z-30">
                     MUTED
                   </div>
              )}
          </div>
      </div>
  );
}

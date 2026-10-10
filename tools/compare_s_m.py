import json

models = ['general', 'n02', 'n03', 'n04', 'n05', 'n06', 'n07', 'n08', 'n10', 'n12', 'n15']
print("=" * 85)
print(f"{'MODEL':<8} | {'METRIC':<20} | {'TIER S (31k)':<14} | {'TIER M (70k)':<14} | {'DELTA':<12}")
print("=" * 85)

summary_rows = []

for m in models:
    s_f = f"reports/{m}.json"
    m_f = f"reports_m/{m}.json"
    s_data = json.load(open(s_f))
    m_data = json.load(open(m_f))

    s_p = s_data['policy']
    m_p = m_data['policy']
    s_v = s_data['value']
    m_v = m_data['value']

    # Unit Move Acc
    u_s = s_p['unit_move_accuracy'] * 100
    u_m = m_p['unit_move_accuracy'] * 100
    du = u_m - u_s

    # Act head acc
    act_s = s_p['heads']['act']['accuracy'] * 100
    act_m = m_p['heads']['act']['accuracy'] * 100
    d_act = act_m - act_s

    # Target head acc
    tgt_s = s_p['heads']['target']['accuracy'] * 100
    tgt_m = m_p['heads']['target']['accuracy'] * 100
    d_tgt = tgt_m - tgt_s

    # Value ranking acc
    vr_s = s_v['ranking_accuracy'] * 100
    vr_m = m_v['ranking_accuracy'] * 100
    d_vr = vr_m - vr_s

    # Value Brier
    vb_s = s_v['brier']
    vb_m = m_v['brier']
    d_vb = vb_m - vb_s

    # Overall joint loss & top-1 by teams
    for s_t, m_t in zip(s_p['by_teams'], m_p['by_teams']):
        t = s_t['teams']
        top1_s = s_t['top1'] * 100
        top1_m = m_t['top1'] * 100
        d_top1 = top1_m - top1_s

        loss_s = s_t['quantized']
        loss_m = m_t['quantized']
        d_loss = loss_m - loss_s

        gain_s = s_t['gain']
        gain_m = m_t['gain']
        d_gain = gain_m - gain_s

        summary_rows.append({
            'model': m,
            'teams': t,
            'unit_acc_s': u_s, 'unit_acc_m': u_m, 'd_unit_acc': du,
            'top1_s': top1_s, 'top1_m': top1_m, 'd_top1': d_top1,
            'loss_s': loss_s, 'loss_m': loss_m, 'd_loss': d_loss,
            'gain_s': gain_s, 'gain_m': gain_m, 'd_gain': d_gain,
            'alpha_s': s_t['alpha'], 'alpha_m': m_t['alpha'],
            'v_brier_s': vb_s, 'v_brier_m': vb_m, 'd_vb': d_vb,
            'v_rank_s': vr_s, 'v_rank_m': vr_m, 'd_vr': d_vr,
            'act_s': act_s, 'act_m': act_m,
            'tgt_s': tgt_s, 'tgt_m': tgt_m,
        })

for r in summary_rows:
    tag = f"{r['model']}(N={r['teams']})"
    print(f"{tag:<14} | Joint Log-Loss       | {r['loss_s']:<14.4f} | {r['loss_m']:<14.4f} | {r['d_loss']:<+10.4f} nats")
    print(f"{'':<14} | Baseline Gain        | {r['gain_s']:<+14.4f} | {r['gain_m']:<+14.4f} | {r['d_gain']:<+10.4f} nats")
    print(f"{'':<14} | Joint Top-1 Acc      | {r['top1_s']:<13.2f}% | {r['top1_m']:<13.2f}% | {r['d_top1']:<+9.2f}%")
    print(f"{'':<14} | Unit Move Acc        | {r['unit_acc_s']:<13.2f}% | {r['unit_acc_m']:<13.2f}% | {r['d_unit_acc']:<+9.2f}%")
    print(f"{'':<14} | Value Ranking Acc    | {r['v_rank_s']:<13.2f}% | {r['v_rank_m']:<13.2f}% | {r['d_vr']:<+9.2f}%")
    print(f"{'':<14} | Value Brier Score    | {r['v_brier_s']:<14.4f} | {r['v_brier_m']:<14.4f} | {r['d_vb']:<+10.4f}")
    print("-" * 85)

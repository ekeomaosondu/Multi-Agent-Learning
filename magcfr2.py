from collections import defaultdict
import matplotlib.pyplot as plt

class Infoset:
    def __init__(self,
                 name='',
                 player='',
                 actions=None,
                 terminal=False,
                 chance=False,
                 payoffs=None,
                 chance_probs=None):
        self.name = name
        self.player = player
        self.actions = actions or []
        self.terminal = terminal
        self.chance = chance
        self.chance_probs = chance_probs or {}
        self.payoffs = payoffs or {}

    def __repr__(self):
        rep = ''
        rep += 'Infoset: ' + self.name + '\n'
        rep += '    Player: ' + self.player + '\n'
        rep += '    Actions: ' + ', '.join(self.actions) + '\n'
        rep += '    Payoffs: ' + ', '.join(str(p) for p in self.payoffs.values())
        if self.chance:
            for a in self.actions:
                rep += f"\n    {a} : {self.chance_probs[a]}"
        return rep + '\n'


class Game:
    def __init__(self, players=None):
        self.players = players or []
        self.hist_to_infoset = {}
        self.infosets = {}
        self.postorder = []

    def read_efg(path: str):
        game = Game()
        with open(path, 'r') as f:
            efg = f.read()

        lines = [line for line in efg.split('\n') if line.strip() != '']
        tree = {}

        for line in lines:
            tokens = line.split(' ')
            history = tokens[1]
            if tokens[0] == 'node':
                if 'player' in tokens:
                    pi = tokens.index('player')
                    player = tokens[pi + 1]
                    if player not in game.players:
                        game.players.append(player)

                if 'terminal' in tokens:
                    payoffs = {}
                    pf = tokens.index('payoffs')
                    for pay in tokens[pf + 1:]:
                        k, v = pay.split('=')
                        payoffs[k] = int(v)
                    infoset = Infoset(history, 'terminal', [], True, False, payoffs)
                    game.infosets[history] = infoset
                    game.hist_to_infoset[history] = infoset

                elif 'chance' in tokens:
                    acts_idx = tokens.index('actions')
                    probs = tokens[acts_idx + 1:]
                    prob_dict = {}
                    for prob in probs:
                        k, v = prob.split('=')
                        prob_dict[k] = float(v)
                    info = Infoset(history, 'chance', list(prob_dict.keys()), False, True, {}, prob_dict)
                    game.infosets[history] = info
                    game.hist_to_infoset[history] = info

                else:
                    player = tokens[3]
                    actions = tokens[5:]
                    tree[history] = (player, actions)

            if tokens[0] == 'infoset':
                nodes_in_infoset = tokens[3:]
                player, actions = tree[tokens[3]]
                info = Infoset(history, player, actions, False, False, {})
                game.infosets[history] = info
                for node in nodes_in_infoset:
                    game.hist_to_infoset[node] = info

        game.postorder = game.node_postorder()
        return game

    def node_postorder(self):
        order = []
        curr = ['/']
        frontier = []
        while curr:
            for node in curr:
                info = self.hist_to_infoset[node]
                if not info.terminal:
                    for a in info.actions:
                        child = self.next_node(node, info.player, a)
                        frontier.append(child)
            order.extend(curr)
            curr, frontier = frontier, []
        return order[::-1]

    def next_node(self, curr_node, player, action):
        if player == 'chance':
            tag = 'C'
        else:
            tag = 'P' + player
        return f'{curr_node}{tag}:{action}/'

    def calc_ev(self, strat1=None, strat2=None, node='/'):
        info = self.hist_to_infoset[node]
        if info.terminal:
            return info.payoffs['1']

        ev = 0.0
        if info.player == 'chance':
            for a in info.actions:
                child = self.next_node(node, 'chance', a)
                ev += info.chance_probs[a] * self.calc_ev(strat1, strat2, child)
        elif info.player == '1':
            if strat1 is None or info.name not in strat1:
                p = 1.0 / len(info.actions)
                for a in info.actions:
                    child = self.next_node(node, '1', a)
                    ev += p * self.calc_ev(strat1, strat2, child)
            else:
                for a, p in strat1[info.name].items():
                    child = self.next_node(node, '1', a)
                    ev += p * self.calc_ev(strat1, strat2, child)
        else:
            if strat2 is None or info.name not in strat2:
                p = 1.0 / len(info.actions)
                for a in info.actions:
                    child = self.next_node(node, '2', a)
                    ev += p * self.calc_ev(strat1, strat2, child)
            else:
                for a, p in strat2[info.name].items():
                    child = self.next_node(node, '2', a)
                    ev += p * self.calc_ev(strat1, strat2, child)
        return ev

    def reach_probs(self, player_i, opp_strat=None):
        probs = {'/': 1.0}
        curr, nxt = ['/'], []
        while curr:
            for node in curr:
                info = self.hist_to_infoset[node]
                if info.terminal:
                    continue
                for a in info.actions:
                    child = self.next_node(node, info.player, a)
                    nxt.append(child)
                    if info.player == 'chance':
                        probs[child] = probs[node] * info.chance_probs[a]
                    elif info.player != player_i:
                        if opp_strat is not None and info.name in opp_strat:
                            probs[child] = probs[node] * opp_strat[info.name].get(a, 0.0)
                        else:
                            probs[child] = probs[node] * (1.0 / len(info.actions))
                    else:
                        probs[child] = probs[node]
            curr, nxt = nxt, []
        return probs

    def _node_values(self, player_i: str, strat1: dict, strat2: dict):
        node_ev = {}
        for node in self.postorder:
            info = self.hist_to_infoset[node]
            if info.terminal:
                node_ev[node] = info.payoffs[player_i]
                continue

            if info.player == 'chance':
                ev = 0.0
                for a in info.actions:
                    child = self.next_node(node, 'chance', a)
                    ev += info.chance_probs[a] * node_ev[child]
                node_ev[node] = ev

            elif info.player == '1':
                ev = 0.0
                if strat1 is not None and info.name in strat1:
                    for a, p in strat1[info.name].items():
                        if p:
                            child = self.next_node(node, '1', a)
                            ev += p * node_ev[child]
                else:
                    p = 1.0 / len(info.actions)
                    for a in info.actions:
                        child = self.next_node(node, '1', a)
                        ev += p * node_ev[child]
                node_ev[node] = ev

            else:
                ev = 0.0
                if strat2 is not None and info.name in strat2:
                    for a, p in strat2[info.name].items():
                        if p:
                            child = self.next_node(node, '2', a)
                            ev += p * node_ev[child]
                else:
                    p = 1.0 / len(info.actions)
                    for a in info.actions:
                        child = self.next_node(node, '2', a)
                        ev += p * node_ev[child]
                node_ev[node] = ev
        return node_ev

    def counterfactual_regret_increments(self, player_i: str, strat1: dict, strat2: dict):
        opp_strat = strat2 if player_i == '1' else strat1
        reach_minus_i = self.reach_probs(player_i, opp_strat)
        node_ev = self._node_values(player_i, strat1, strat2)

        regrets = {}
        for node in self.postorder:
            info = self.hist_to_infoset[node]
            if info.terminal or info.player != player_i:
                continue

            v_here = node_ev[node]
            pi_minus_i = reach_minus_i.get(node, 0.0)
            if pi_minus_i == 0.0:
                continue

            if info.name not in regrets:
                regrets[info.name] = {a: 0.0 for a in info.actions}

            for a in info.actions:
                child = self.next_node(node, player_i, a)
                v_child = node_ev[child]
                regrets[info.name][a] += pi_minus_i * (v_child - v_here)

        return regrets

    def tree_values(self, player_i, opp_player, strategy=None):
        response = {}
        reach = self.reach_probs(player_i, strategy)
        policy_ev_table = {} 
        infoset_ev_table = {}  
        node_ev_table = {}   

        for node in self.postorder:
            info = self.hist_to_infoset[node]
            if info.terminal:
                infoset_ev_table[info.name] = info.payoffs[player_i]
                node_ev_table[node] = info.payoffs[player_i]
                continue

            if info.player == player_i:
                if info.name not in policy_ev_table:
                    policy_ev_table[info.name] = {a: (0.0, 0.0) for a in info.actions}

                for a in info.actions:
                    child = self.next_node(node, info.player, a)
                    num, den = policy_ev_table[info.name][a]
                    policy_ev_table[info.name][a] = (num + node_ev_table.get(child, 0.0) * reach.get(child, 0.0),
                                                     den + reach.get(child, 0.0))

            else:
                node_ev = 0.0
                for a in info.actions:
                    child = self.next_node(node, info.player, a)
                    child_info = self.hist_to_infoset[child]

                    if not child_info.terminal and child_info.player == player_i:
                        if child_info.name not in infoset_ev_table:
                 
                            optimal_value = float('-inf')
                            optimal_move = child_info.actions[0]
                            for ca in child_info.actions:
                                n, d = policy_ev_table[child_info.name][ca]
                                value = (n / d) if d != 0 else float('-inf')
                                if value > optimal_value:
                                    optimal_value = value
                                    optimal_move = ca
                            infoset_ev_table[child_info.name] = optimal_value
                            response[child_info.name] = (optimal_move, optimal_value)

                        
                        best_a = response[child_info.name][0]
                        node_ev_table[child] = node_ev_table[self.next_node(child, child_info.player, best_a)]

                    if info.player == opp_player:
                        if strategy is not None and info.name in strategy:
                            node_ev += strategy[info.name].get(a, 0.0) * node_ev_table.get(child, 0.0)
                        else:
                            node_ev += (1.0 / len(info.actions)) * node_ev_table.get(child, 0.0)
                    else:
  
                        if info.name not in infoset_ev_table:
                            infoset_ev_table[info.name] = {aa: 0.0 for aa in info.actions}
                        if child_info.player == opp_player:
                            infoset_ev_table[info.name][a] += node_ev_table.get(child, 0.0)
                        else:
                            infoset_ev_table[info.name][a] += infoset_ev_table.get(child_info.name, 0.0)
                        if child_info.player == player_i:
                            best_a = response[child_info.name][0]
                            node_ev_table[child] = node_ev_table[self.next_node(child, child_info.player, best_a)]
                        node_ev += info.chance_probs[a] * node_ev_table.get(child, 0.0)

                node_ev_table[node] = node_ev

        for I in policy_ev_table:
            for a in policy_ev_table[I]:
                n, d = policy_ev_table[I][a]
                policy_ev_table[I][a] = (n / d) if d != 0 else float('-inf')

        return policy_ev_table

    def best_response(self, player_i, strategy=None):
        response = {}
        opp_player = '2' if player_i == '1' else '1'
        policy_ev_table = self.tree_values(player_i, opp_player, strategy)
        for info in self.infosets.values():
            if info.player == player_i and not info.terminal:
                best_a, best_v = None, float('-inf')
                for a in info.actions:
                    v = policy_ev_table[info.name][a]
                    if v > best_v:
                        best_a, best_v = a, v
                if best_a is not None:
                    response[info.name] = {best_a: 1.0}
        return response

    def equilibrium_gap(self, strat1=None, strat2=None):
        return self.calc_ev(self.best_response('1', strat2), strat2) - \
               self.calc_ev(strat1, self.best_response('2', strat1))


def uniform_strategy(game: Game, player: str):
    out = {}
    for name, info in game.infosets.items():
        if info.terminal or info.player != player:
            continue
        n = len(info.actions)
        out[name] = {a: 1.0/n for a in info.actions}
    return out

def own_reach_probs(game: Game, player_i: str, my_strat: dict):
    probs = {'/': 1.0}
    curr, nxt = ['/'], []
    while curr:
        for node in curr:
            info = game.hist_to_infoset[node]
            if info.terminal:
                continue
            for a in info.actions:
                child = game.next_node(node, info.player, a)
                nxt.append(child)
                if info.player == player_i:
                    if my_strat is not None and info.name in my_strat:
                        p = my_strat[info.name].get(a, 0.0)
                        probs[child] = probs[node] * p
                    else:
                        probs[child] = probs[node] * (1.0 / len(info.actions))
                else:
                    probs[child] = probs[node]
        curr, nxt = nxt, []
    return probs

def accumulate_avg_counts(game: Game, player: str, my_strat: dict, avg_counts: dict, denom_counts: dict):
    pi_i = own_reach_probs(game, player, my_strat)
    infoset_ownreach = defaultdict(float)
    for node, prob in pi_i.items():
        info = game.hist_to_infoset[node]
        if info.terminal or info.player != player:
            continue
        infoset_ownreach[info.name] += prob

    for I, info in game.infosets.items():
        if info.terminal or info.player != player:
            continue
        reach_I = infoset_ownreach.get(I, 0.0)
        if reach_I == 0.0:
            continue
        if I not in avg_counts:
            avg_counts[I] = defaultdict(float)
        denom_counts[I] = denom_counts.get(I, 0.0) + reach_I
        for a, p in my_strat.get(I, {}).items():
            avg_counts[I][a] += reach_I * p

def normalize_avg(avg_counts: dict, denom_counts: dict):
    avg_pol = {}
    for I, counts in avg_counts.items():
        denom = denom_counts.get(I, 0.0)
        if denom <= 0.0:
            continue
        avg_pol[I] = {a: (counts[a]/denom) for a in counts}
        s = sum(avg_pol[I].values())
        if s > 0:
            for a in avg_pol[I]:
                avg_pol[I][a] /= s
    return avg_pol

class RegMin:
    def __init__(self, actions, infoset_name):
        self.infoset = infoset_name
        self.cum_reg = {a: 0.0 for a in actions}
        self.strat = self.next_strategy()

    def next_strategy(self):
        pos = {a: max(0.0, r) for a, r in self.cum_reg.items()}
        s = sum(pos.values())
        if s == 0.0:
            n = len(self.cum_reg)
            self.strat = {a: 1.0/n for a in self.cum_reg}
        else:
            self.strat = {a: pos[a]/s for a in self.cum_reg}
        return self.strat

    def observe_advantages(self, deltas: dict):
        for a, d in deltas.items():
            self.cum_reg[a] += d

class CFR_P1_vs_UniformP2:
    def __init__(self, game: Game):
        self.game = game
        self.p1, self.p2 = '1', '2'
        self.minim = {
            name: RegMin(info.actions, name)
            for name, info in self.game.infosets.items()
            if not info.terminal and info.player == self.p1
        }
        self.avg_counts = {}
        self.denom_counts = {}
        self.uniform_p2 = uniform_strategy(game, self.p2)

    def current_p1(self):
        return {I: self.minim[I].strat for I in self.minim}

    def step(self):
        s1 = self.current_p1()
        s2 = self.uniform_p2

        adv = self.game.counterfactual_regret_increments(self.p1, s1, s2)
        for I, d in adv.items():
            if I in self.minim:
                self.minim[I].observe_advantages(d)

        for I in self.minim:
            self.minim[I].next_strategy()

        s1 = self.current_p1()
        accumulate_avg_counts(self.game, self.p1, s1, self.avg_counts, self.denom_counts)

    def average_p1(self):
        return normalize_avg(self.avg_counts, self.denom_counts)


class CFR_TwoPlayer:
    def __init__(self, game: Game):
        self.game = game
        self.p1, self.p2 = '1', '2'
        self.minim1 = {
            name: RegMin(info.actions, name)
            for name, info in self.game.infosets.items()
            if not info.terminal and info.player == self.p1
        }
        self.minim2 = {
            name: RegMin(info.actions, name)
            for name, info in self.game.infosets.items()
            if not info.terminal and info.player == self.p2
        }
        self.avg1_counts, self.den1 = {}, {}
        self.avg2_counts, self.den2 = {}, {}

    def cur1(self): return {I: self.minim1[I].strat for I in self.minim1}
    def cur2(self): return {I: self.minim2[I].strat for I in self.minim2}

    def step(self):
        s1, s2 = self.cur1(), self.cur2()

        adv1 = self.game.counterfactual_regret_increments('1', s1, s2)
        for I, d in adv1.items():
            if I in self.minim1:
                self.minim1[I].observe_advantages(d)

        adv2 = self.game.counterfactual_regret_increments('2', s1, s2)
        for I, d in adv2.items():
            if I in self.minim2:
                self.minim2[I].observe_advantages(d)

        for I in self.minim1: self.minim1[I].next_strategy()
        for I in self.minim2: self.minim2[I].next_strategy()

        s1, s2 = self.cur1(), self.cur2()
        accumulate_avg_counts(self.game, '1', s1, self.avg1_counts, self.den1)
        accumulate_avg_counts(self.game, '2', s2, self.avg2_counts, self.den2)

    def avg_strats(self):
        return normalize_avg(self.avg1_counts, self.den1), normalize_avg(self.avg2_counts, self.den2)


rps_path   = '/Users/ekeomaosondu/Desktop/MIT 2027/Fall 2025/Multi-Agent/efgs/rock_paper_superscissors.txt'
kuhn_path  = '/Users/ekeomaosondu/Desktop/MIT 2027/Fall 2025/Multi-Agent/efgs/kuhn.txt'
leduc_path = '/Users/ekeomaosondu/Desktop/MIT 2027/Fall 2025/Multi-Agent/efgs/leduc2.txt'

rps   = Game.read_efg(rps_path)
kuhn  = Game.read_efg(kuhn_path)
leduc = Game.read_efg(leduc_path)

def run_p1_vs_uniform(game, T=1000, title=''):
    cfr = CFR_P1_vs_UniformP2(game)
    ev_traj = []
    unif2 = cfr.uniform_p2
    for t in range(1, T+1):
        cfr.step()
        xbar = cfr.average_p1()
        ev = game.calc_ev(xbar, unif2)
        ev_traj.append(ev)
    plt.figure()
    plt.plot(range(1, T+1), ev_traj)
    plt.xlabel('Iterations T')
    plt.ylabel('EV for Player 1 ((u1(x)_T, y_T))')
    plt.title(title or 'P1 vs Uniform P2')
    plt.grid(True)
    return ev_traj

T = 1000
run_p1_vs_uniform(rps,   T=T, title='RPS: P1 vs Uniform P2')
run_p1_vs_uniform(kuhn,  T=T, title='Kuhn: P1 vs Uniform P2')
run_p1_vs_uniform(leduc, T=T, title='Leduc: P1 vs Uniform P2')

def run_two_player_cfr(game, T=1000, title_prefix=''):
    cfr = CFR_TwoPlayer(game)
    util_traj, gap_traj = [], []
    for t in range(1, T+1):
        cfr.step()
        xbar, ybar = cfr.avg_strats()
        util = game.calc_ev(xbar, ybar)
        gap  = game.equilibrium_gap(xbar, ybar)
        util_traj.append(util)
        gap_traj.append(gap)

    plt.figure()
    plt.plot(range(1, T+1), util_traj)
    plt.xlabel('Iterations T')
    plt.ylabel('u1(x̄_T, ȳ_T)')
    plt.title(f'{title_prefix} Utility of Average Strategies')
    plt.grid(True)

    plt.figure()
    plt.plot(range(1, T+1), gap_traj)
    plt.xlabel('Iterations T')
    plt.ylabel('Average Strategy Exploitability')
    plt.title(f'{title_prefix} NE Gap of Average Strategies')
    plt.grid(True)

    return util_traj, gap_traj

run_two_player_cfr(rps,   T=T, title_prefix='RPS')
run_two_player_cfr(kuhn,  T=T, title_prefix='Kuhn')
run_two_player_cfr(leduc, T=T, title_prefix='Leduc')

plt.show()

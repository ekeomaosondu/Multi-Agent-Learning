class Infoset:
        def __init__(self,
                     name = '', 
                     player = '', 
                     actions = [], 
                     terminal = False, 
                     chance = False, 
                     payoffs = {},
                     chance_probs = {}):
            
            self.name = name
            self.player = player
            self.actions = actions
            self.terminal = terminal
            self.chance = chance
            self.chance_probs = chance_probs
            self.payoffs = payoffs

        def __repr__(self):
            rep = ''
            rep += 'Infoset: ' + self.name + '\n'
            rep += '    Player: ' + self.player + '\n'
            rep += '    Actions: '
            for action in self.actions:
                rep += action + ", "
            rep += '\n'
            rep += '    Payoffs: '
            for payoff in self.payoffs.values():
                rep += str(payoff) + ", "
            if self.chance:
                for action in self.actions:
                    rep +=  "\n    " + action + " : " + str(self.chance_probs[action])
            return rep + '\n'

class Game:

    def __init__(self, players: list = [], ):
        self.players = players
        self.hist_to_infoset = {}
        self.infosets = {}

    def read_efg(path: str):
        game = Game()
        try:
            with open(path, 'r') as file:
                efg = file.read()
        except:
            raise Exception()

        lines = [line for line in efg.split('\n') if line != '']

        tree = {}
        for line in lines:
            tokens = line.split(' ')
            history = tokens[1]
            if tokens[0] == 'node':
                if 'player' in tokens:
                    player_ind = tokens.index('player')
                    player = tokens[player_ind + 1]
                    if player not in game.players:
                        game.players.append(player)
            
                if 'terminal' in tokens:
                    payoffs = {}
                    payoff_ind = tokens.index('payoffs')
                    for payoff in tokens[payoff_ind + 1:]:
                        payoff = payoff.split('=')
                        payoffs[payoff[0]] = int(payoff[1])
                    
                    infoset = Infoset(history, 'terminal', [], True, False, payoffs)
                    game.infosets[history] = infoset
                    game.hist_to_infoset[history] = infoset
                elif 'chance' in tokens:
                    probs = tokens[tokens.index('actions') + 1:]
                    prob_dict = {}
                    for prob in probs:
                        tokenized = prob.split('=')
                        prob_dict[tokenized[0]] = float(tokenized[1])

                    game.infosets[history] = Infoset(history, 
                                                     'chance', 
                                                     list(prob_dict.keys()), 
                                                     False, 
                                                     True, 
                                                     {},
                                                     prob_dict)

                    game.hist_to_infoset[history] = game.infosets[history]

                
                else:
                    player = tokens[3]
                    tree[history] = player, tokens[5:]
                

            if tokens[0] == 'infoset':
                nodes_in_infoset = tokens[3:]
                
                player, actions = tree[tokens[3]]

                game.infosets[history] = Infoset(history, player, actions, False, {})
                for node in nodes_in_infoset:
                    game.hist_to_infoset[node] = game.infosets[history]

        game.postorder = game.node_postorder()
        return game

    def trace(self, name, player):
        infoset = self.hist_to_infoset[name]
        colon = name.rfind(':')
        if name == '/' or name == '':
            return []
        elif name[colon - 1] != player:
            name = name[:-1]
            last_slash = name.rfind('/')
            return self.trace(name[:last_slash + 1], player)
        else:
            name = name[:-1]
            last_slash = name.rfind('/')
            last_action = name[:last_slash].rfind('/')
            colon = name.rfind(':')
            return self.trace(name[:last_slash + 1], player) + [(name[:last_slash + 1], name[colon + 1:])]
    
    def reach_probs(self, player_i, opp_strat=None):
        probs = {'/': 1}
        curr = ['/']
        next_frontier = []
        while curr:
            for node in curr:
                infoset = self.hist_to_infoset[node]
                if not infoset.terminal:
                    for action in infoset.actions:
                        child = self.next_node(node, infoset.player, action)
                        next_frontier.append(child)
                        if infoset.player == 'chance':
                            probs[child] = probs[node] * infoset.chance_probs[action]
                        elif infoset.player != player_i:
                            if opp_strat is not None:
                                probs[child] = probs[node] * opp_strat[infoset.name][action]
                            else:
                                probs[child] = probs[node] * 1/len(infoset.actions)
                        else:
                            probs[child] = probs[node]
                        next_frontier.append(child)

            curr, next_frontier = next_frontier, []
        return probs
        
    def node_postorder(self):
        order = []

        curr = ['/']
        frontier = []

        while curr:
            for node in curr:
                infoset = self.hist_to_infoset[node]
                if not infoset.terminal:        
                    for action in infoset.actions:
                        child = self.next_node(node, infoset.player, action)
                        frontier.append(child)

            order.extend(curr)
            curr, frontier = frontier, []

        return order[::-1]
        
    def next_node(self, curr_node, player, action):
        if player == 'chance':
            player_tag = 'C'

        else:
            player_tag = 'P' + player
        return f'{curr_node}{player_tag}:{action}/'

    def best_response(self, player_i, strategy=None):
        response = {}

        opp_player = '2' if player_i == '1' else '1'
        policy_ev_table = self.tree_values(player_i, opp_player, strategy)
        for infoset in self.infosets.values():
            if infoset.player == player_i:
                for action in infoset.actions: 
                    value = policy_ev_table[infoset.name][action]
                    if infoset.name not in response or value > response[infoset.name][1]:
                        response[infoset.name] = (action, value)

        reach = self.reach_probs(player_i, strategy)

        return {key: {value[0]: 1} for key, value in response.items()}

    def tree_values(self, player_i, opp_player, strategy=None):
        response = {}
        reach = self.reach_probs(player_i, strategy)
        policy_ev_table = {} #maps from infoset to {action: ev}
        infoset_ev_table = {} #maps from infoset to ev of being in the state
        node_ev_table = {} #maps from particular game tree nodes to ev
        #in reverse topo of nodes:
        for node in self.node_postorder():
            infoset = self.hist_to_infoset[node]
            if infoset.terminal:
                infoset_ev_table[infoset.name] = infoset.payoffs[player_i]
                node_ev_table[node] = infoset.payoffs[player_i]

            elif infoset.player == player_i:
                if infoset.name not in policy_ev_table:
                    policy_ev_table[infoset.name] = {}
                    for action in infoset.actions:
                        policy_ev_table[infoset.name][action] = 0, 0

                for action in infoset.actions:
                    child = self.next_node(node, infoset.player, action)
                    child_infoset = self.hist_to_infoset[child]
                    
                    policy_ev_table[infoset.name][action] = policy_ev_table[infoset.name][action][0] + node_ev_table[child] * reach[child], policy_ev_table[infoset.name][action][1] + reach[child]

            else:
                node_ev = 0
                for action in infoset.actions:
                    child = self.next_node(node, infoset.player, action)
                    child_infoset = self.hist_to_infoset[child]

                    if not child_infoset.terminal and child_infoset.player == player_i:
                        if child_infoset.name not in infoset_ev_table:
                            
                            optimal_value = float('-inf')
                            optimal_move = None
                            for child_action in child_infoset.actions:
                                value = None
                                if policy_ev_table[child_infoset.name][child_action][1] != 0:
                                    value = policy_ev_table[child_infoset.name][child_action][0] / policy_ev_table[child_infoset.name][child_action][1]
                                if value > optimal_value:
                                    optimal_value = value
                                    optimal_move = child_action
                            
                            infoset_ev_table[child_infoset.name] = optimal_value
                            response[child_infoset.name] = optimal_move, optimal_value

                        node_ev_table[child] = node_ev_table[self.next_node(child, child_infoset.player, response[child_infoset.name][0])] 

                    if infoset.player == opp_player:
                        if strategy is not None:
                            node_ev += strategy[infoset.name][action] * node_ev_table[child]
                        else:
                            node_ev += 1/len(infoset.actions) * node_ev_table[child]
                    else:
                        if infoset.name not in infoset_ev_table:
                            infoset_ev_table[infoset.name] = {}
                            for action in infoset.actions:
                                infoset_ev_table[infoset.name][action] = 0

                        if child_infoset.player == opp_player:
                            infoset_ev_table[infoset.name][action] += node_ev_table[child] 
                        else:
                            infoset_ev_table[infoset.name][action] += infoset_ev_table[child_infoset.name]
                        if child_infoset.player == player_i:
                            node_ev_table[child] = node_ev_table[self.next_node(child, child_infoset.player, response[child_infoset.name][0])]

                        node_ev += infoset.chance_probs[action] * node_ev_table[child]
                    node_ev_table[node] = node_ev

        for infoset in policy_ev_table:
            for action in policy_ev_table[infoset]:
                if policy_ev_table[infoset][action][1] != 0:
                    policy_ev_table[infoset][action] = policy_ev_table[infoset][action][0] / policy_ev_table[infoset][action][1]
                else:
                    policy_ev_table[infoset][action] = float('-inf')

        return policy_ev_table

    def calc_ev(self, strat1 = None, strat2 = None, node = '/'):
        ev = 0
        infoset = self.hist_to_infoset[node]
        if not infoset.terminal:
            for action in infoset.actions:
                child = self.next_node(node, infoset.player, action)
                child_infoset = self.hist_to_infoset[child]
                if infoset.player == 'chance':
                    ev += infoset.chance_probs[action] * self.calc_ev(strat1, strat2, child)
                elif infoset.player == '1':
                    if strat1 is not None:
                        if action in strat1[infoset.name]:
                            ev += strat1[infoset.name][action] * self.calc_ev(strat1, strat2, child)
                    else:
                        ev += 1/len(infoset.actions) * self.calc_ev(strat1, strat2, child)
                else:
                    if strat2 is not None:
                        if action in strat2[infoset.name]:
                            ev += strat2[infoset.name][action] * self.calc_ev(strat1, strat2, child)
                    else:
                        ev += 1/len(infoset.actions) * self.calc_ev(strat1, strat2, child)
        else:
            ev = infoset.payoffs['1']

        return ev
    
    def equilibrium_gap(self, strat1 = None, strat2 = None):
        return self.calc_ev(self.best_response('1', strat2), strat2) - self.calc_ev(strat1, self.best_response('2', strat1))

    def _node_values(self, player_i: str, strat1: dict | None, strat2: dict | None):
        """Return node_ev: map node->EV for player_i under (strat1,strat2)."""
        node_ev = {}
        for node in self.node_postorder():
            infoset = self.hist_to_infoset[node]
            if infoset.terminal:
                node_ev[node] = infoset.payoffs[player_i]
                continue

            if infoset.player == 'chance':
                ev = 0.0
                for a in infoset.actions:
                    child = self.next_node(node, 'chance', a)
                    ev += infoset.chance_probs[a] * node_ev[child]
                node_ev[node] = ev
            elif infoset.player == '1':
                ev = 0.0
                if strat1 is not None and infoset.name in strat1:
                    pol = strat1[infoset.name]
                    for a in infoset.actions:
                        p = pol.get(a, 0.0)
                        if p:
                            child = self.next_node(node, '1', a)
                            ev += p * node_ev[child]
                else:
                    # uniform if no strategy provided
                    p = 1.0 / len(infoset.actions)
                    for a in infoset.actions:
                        child = self.next_node(node, '1', a)
                        ev += p * node_ev[child]
                node_ev[node] = ev
            else:  # infoset.player == '2'
                ev = 0.0
                if strat2 is not None and infoset.name in strat2:
                    pol = strat2[infoset.name]
                    for a in infoset.actions:
                        p = pol.get(a, 0.0)
                        if p:
                            child = self.next_node(node, '2', a)
                            ev += p * node_ev[child]
                else:
                    p = 1.0 / len(infoset.actions)
                    for a in infoset.actions:
                        child = self.next_node(node, '2', a)
                        ev += p * node_ev[child]
                node_ev[node] = ev
        return node_ev

class CFR():
    def __init__(self, game, player_i):
        self.game = game
        self.player_i = player_i
        self.opp_player = '2' if player_i == '1' else '1'
        self.minimizers = {} #map from info sets to reg min
        for infoset in self.game.infosets:
            if self.game.infosets[infoset].player == player_i:
                self.minimizers[infoset] = RegMin(self.game.infosets[infoset].actions, infoset)
        self.children, self.parent = self.transition()

    def transition(self):
        parent = {'/': None}
        children = {}
        curr = ['/']
        next_frontier = []

        while curr:
            for node in curr:
                infoset = self.game.hist_to_infoset[node]
                if infoset.name not in children:
                    children[infoset.name] = {}
                if not infoset.terminal:
                    for action in infoset.actions:
                        child = self.game.next_node(node, infoset.player, action)
                        child_infoset = self.game.hist_to_infoset[child]

                        if child_infoset.name not in parent:
                            next_frontier.append(child)
                            parent[child_infoset.name] = node
                            children[infoset.name][action] = child_infoset.name

                else:
                    children[infoset.name] = {None: None}

            curr = next_frontier
            next_frontier = []

        return children, parent

    def next_strategy(self):
        strat = {}
        for node in self.game.node_postorder()[::-1]:
            infoset = self.game.hist_to_infoset[node]

            if infoset.name not in strat:
                if infoset.player == self.player_i:
                    strat[infoset.name] = {}
                    j = self.game.infosets[infoset.name]
                    b = self.minimizers[j.name].next_strategy()
                    for action in j.actions:
                        tr = self.game.trace(node, self.player_i)
                        if len(tr) == 0:
                            strat[infoset.name][action] = b[action]
                        else:
                            parent, move = tr[-1]
                            strat[infoset.name][action] = b[action] * strat[self.game.hist_to_infoset[parent].name][move]

        return strat

    def observe_utility(self, gradient):
        strat = self.next_strategy()
        V = {}
        for node in self.game.node_postorder():
            infoset = self.game.hist_to_infoset[node]
            if infoset.terminal:
                V[infoset.name] = 0
            if infoset.player == self.player_i:
                local_strat = self.minimizers[infoset.name].strat
                V[infoset.name] = 0 
                for action in local_strat:
                    child = self.game.next_node(node, infoset.player, action)
                    print(child, infoset.name)
                    child_infoset = self.game.hist_to_infoset[child]
                    V[infoset.name] += local_strat[action] * (gradient[infoset.name][action] + V[child])
            else:
                V[infoset.name] = 0
                for action in infoset.actions:
                    child = self.game.next_node(node, infoset.player, action)
                    child_infoset = self.game.hist_to_infoset[child]
                    V[infoset.name] += V[child]

        g = {}
        for node in self.game.node_postorder():
            infoset = self.game.hist_to_infoset[node]
            if infoset.player == self.player_i:
                for action in actions:
                    child = self.game.next_node(node, infoset.player, action)
                    child_infoset = self.game.hist_to_infoset[child]
                    g[infoset.name][action] = gradient[infoset.name][action] + V[child][action]

                self.minimizers[infoset.name].observe_utility(g[infoset.name]) 
class RegMin():
    def __init__(self, actions, infoset):
        self.infoset = infoset #infoset name
        self.cum_reg = {a: 0 for a in actions}
        self.strat = self.next_strategy()

    def next_strategy(self):
        if sum(max(0, value) for value in self.cum_reg.values()) == 0:
            self.strat = {action: 1/len(self.cum_reg) for action in self.cum_reg}
        else:
            self.strat = {action: max(value, 0) / sum(max(0, value) for value in self.cum_reg.values()) for action, value in self.cum_reg.items()}
        return self.strat

    def observe_utility(self, utility):
        self.regret = {action: max(regret + utility[action] - (utility[action] * self.strat[action]), 0) for action, regret in self.regret.items()}
        
rps = '/Users/ekeomaosondu/Desktop/MIT 2027/Fall 2025/Multi-Agent/efgs/rock_paper_superscissors.txt'
kuhn = '/Users/ekeomaosondu/Desktop/MIT 2027/Fall 2025/Multi-Agent/efgs/kuhn.txt'
leduc = '/Users/ekeomaosondu/Desktop/MIT 2027/Fall 2025/Multi-Agent/efgs/leduc2.txt'
rps = Game.read_efg(rps)
kuhn = Game.read_efg(kuhn)
leduc = Game.read_efg(leduc)






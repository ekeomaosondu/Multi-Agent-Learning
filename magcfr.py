from email import policy
from logging import info


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
                    
                    
                    infoset = Infoset(history, 'terminal', [''], True, True, payoffs)
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
        
        return game
    
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
                    value = policy_ev_table[infoset.name][action][0] / policy_ev_table[infoset.name][action][1] if policy_ev_table[infoset.name][action][1] != 0 else float('-inf')
                    if infoset.name not in response or value > response[infoset.name][1]:
                        response[infoset.name] = (action, value)

        return response

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
        return policy_ev_table

    def cfr(game):
        def next_strategy():
            pass

        pass
            

rps = '/Users/ekeomaosondu/Desktop/MIT 2027/Fall 2025/Multi-Agent/efgs/rock_paper_superscissors.txt'
kuhn = '/Users/ekeomaosondu/Desktop/MIT 2027/Fall 2025/Multi-Agent/efgs/kuhn.txt'
leduc = '/Users/ekeomaosondu/Desktop/MIT 2027/Fall 2025/Multi-Agent/efgs/leduc2.txt'
rps = Game.read_efg(rps)
kuhn = Game.read_efg(kuhn)
leduc = Game.read_efg(leduc)

uniform_strategy = {
    '/P1:?/': {'r':1/3, 'p':1/3, 's':1/3},
}

game = rps

for key, value in game.best_response('2').items(): #infoset EV
    print(key, value)
print()
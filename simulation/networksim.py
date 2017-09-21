from distributions import (
    transform,
    normal_distribution,
)
import random


class NetworkSimulator():

    def __init__(self, latency=50, reliability=0.9):
        self.agents = []
        self.latency_distribution_sample = transform(normal_distribution(latency, (latency * 2) // 5), lambda x: max(x, 0))
        # 1 unit = 0.01 sec
        self.time = 0
        self.objqueue = {}
        self.peers = {}
        self.reliability = reliability

    def clear_peers(self, network_id=1):
        self.peers[network_id] = {}

    def add_peers(self, agent, num_peers=5, network_id=1, peer_list=None):
        if peer_list is None:
            peer_list = self.agents
        p = []
        if len(peer_list) == 1:
            return
        while len(p) <= num_peers // 2:
            p.append(random.choice(peer_list))
            if p[-1] == agent and (
                    agent.id not in self.peers[network_id] or p[-1] not in self.peers[network_id][agent.id]):
                p.pop()
        self.peers[network_id][agent.id] = self.peers[network_id].get(agent.id, []) + p
        self.peers[network_id][agent.id] = list(set(self.peers[network_id][agent.id]))
        for peer in p:
            self.peers[network_id][peer.id] = self.peers[network_id].get(peer.id, []) + [agent]
            self.peers[network_id][peer.id] = list(set(self.peers[network_id][peer.id]))
        print('Agent [V {}] has peers:{} in network_id {}'.format(
            agent.id, [v.id for v in self.peers[network_id][agent.id]], network_id))

    def remove_peer(self, agent, peer, network_id=1):
        if peer.id in self.peers[network_id][agent.id]:
            self.peers[network_id][agent.id].remove(peer.id)
        if agent.id in self.peers[network_id][peer.id]:
            self.peers[network_id][peer.id].remove(agent.id)

    def get_peers(self, agent, network_id):
        try:
            return self.peers[network_id][agent.id]
        except KeyError:
            return []

    def generate_peers(self, num_peers=5, network_id=1):
        self.clear_peers(network_id)
        for a in self.agents:
            self.add_peers(a, num_peers=num_peers, network_id=network_id)

    def tick(self):
        if self.time in self.objqueue:
            for recipient, obj, network_id, sender in self.objqueue[self.time]:
                if random.random() < self.reliability:
                    recipient.on_receive(obj, network_id, sender)
            del self.objqueue[self.time]
        for a in self.agents:
            a.tick()
        self.time += 1

    def run(self, steps):
        for i in range(steps):
            self.tick()

    def broadcast(self, sender, obj, additional_latency=0, network_id=1):
        # recv_time = self.time + self.latency_distribution_sample() + additional_latency
        # print('[V {}] broadcasts object, now is {}, recv_time about {} '.format(sender, self.time, recv_time))
        try:
            if sender.id in self.peers[network_id]:
                for p in self.peers[network_id][sender.id]:
                    recv_time = self.time + self.latency_distribution_sample() + additional_latency
                    if recv_time not in self.objqueue:
                        self.objqueue[recv_time] = []
                    self.objqueue[recv_time].append((p, obj, network_id, sender.id))
            else:
                print('[broadcast] V {} has no peer'.format(sender.id))
        except KeyError:
            print('network_id: {}'.format(network_id))
            print('sender.id: {}'.format(sender.id))
            raise

    def direct_send(self, sender, to_id, obj, network_id=1):
        for a in self.agents:
            if a.id == to_id:
                recv_time = self.time + self.latency_distribution_sample()
                if recv_time not in self.objqueue:
                    self.objqueue[recv_time] = []
                self.objqueue[recv_time].append((a, obj, network_id, sender.id))

    def knock_offline_random(self, n, network_id=1):
        ko = {}
        while len(ko) < n:
            c = random.choice(self.agents)
            ko[c.id] = c
        for c in ko.values():
            self.peers[network_id][c.id] = []
        for a in self.agents:
            self.peers[network_id][a.id] = [x for x in self.peers[network_id][a.id] if x.id not in ko]

    def partition(self, network_id=1):
        a = {}
        while len(a) < len(self.agents) / 2:
            c = random.choice(self.agents)
            a[c.id] = c
        for c in self.agents:
            if c.id in a:
                self.peers[network_id][c.id] = [x for x in self.peers[network_id][c.id] if x.id in a]
            else:
                self.peers[network_id][c.id] = [x for x in self.peers[network_id][c.id] if x.id not in a]

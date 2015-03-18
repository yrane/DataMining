###############################################################################
# The 8-Puzzle problem: Implement the classic 8-puzzle problem using A* search
# Submitted by: Yogesh Rane(yrane2)
# Date: 02/2/2015
###############################################################################


import heapq
import random
import matplotlib
import pylab

def solve(start, finish, heuristic):
    """Find the shortest path from Start to Finish"""
    heap = []

    h = {} # heuristic function cache
    g = {} # shortest path to a node

    g[start] = 0
    h[start] = 0

    heapq.heappush(heap, (0, 0, start))
    for k in xrange(1000000):
        a,b, current = heapq.heappop(heap)
        if current == finish:
            # print "Puzzle Solved!"
            # print current.showMatrix()
            print "Steps:", int(g[current]), "Nodes Expanded:", k
            return g[current], k

        moves = current.get_moves()
        distance = g[current]
        for mv in moves:
            if mv not in g or g[mv] > distance + 1:
                g[mv] = distance + 1
                if mv not in h:
                    h[mv] = heuristic(mv)
                heapq.heappush(heap, (g[mv] + h[mv], k, mv))
    else:
        raise Exception("did not find a solution")

def shuffle(pos, n):
    """ Get possible moves and then choose one of the moves"""
    for k in xrange(n):
        state = list(pos.get_moves())
        pos = random.choice(state)
    return pos


def misplaced_heuristic(pos):
    """The number of tiles out of place."""
    n2 = pos.n2
    count = 0
    for k in xrange(n2):
        if pos.state[k] != k:
            count += 1
    return count

def distance_heuristic(pos):
    """ The Manhattan distance heuristic"""
    n = pos.n
    def row(x): return x / n
    def col(x): return x % n
    score = 0
    for idx, x in enumerate(pos.state):
        if x == 0: continue
        ir,ic = row(idx), col(idx)
        xr,xc = row(x), col(x)
        score += abs(ir-xr) + abs(ic-xc)
    return score


def gaschnig_heuristic(pos):
#     n = position.n
    moves = 0
    ind = 0
    temp = [None]*9
    for num in pos.state:
        temp[ind] = num
        ind += 1

    while temp != [0,1,2,3,4,5,6,7,8]:
        if temp[0] == 0:
            i = 1
            while temp[i] == i:
                i += 1
            var = temp[i]
            temp[i] = 0
            temp[0] = var
#             swap blank with any mismatch
        else:
            i = 1
            while temp[i] != 0:
                i += 1
            j = 0
            while temp[j] != i:
                j += 1
            temp[j] = 0
            temp[i] = i
#             swap blank with matched tile
        moves += 1
    return moves

class PuzzleBlock(object):
    def __init__(self, n, state=None):
        """Create an nxn block puzzle

        Use state to initialize to a specific state.
        """
        self.n = n
        self.n2 = n * n
        if state is None:
            self.state = [(x) % self.n2 for x in xrange(self.n2)]
        else:
            self.state = list(state)
        self.hsh = None
        self.last_move = []

    def __hash__(self):
        if self.hsh is None:
            self.hsh = hash(tuple(self.state))
        return self.hsh

    def __repr__(self):
        return "PuzzleBlock(%d, %s)" % (self.n, self.state)

    def showMatrix(self):
        ys = ["%2d" % x for x in self.state]
        state = [" ".join(ys[k:k+self.n]) for k in xrange(0,self.n2, self.n)]
        return "\n".join(state)

    def __eq__(self, other):
        return self.state == other.state

    def get_moves(self):
        # Find the 0 tile, and then generate any moves we
        # can by sliding another block into its place.
        tile0 = self.state.index(0)
        def swap(i):
            j = tile0
            tmp = list(self.state)
            last_move = tmp[i]
            tmp[i], tmp[j] = tmp[j], tmp[i]
            result = PuzzleBlock(self.n, tmp)
            result.last_move = last_move
            return result

        if tile0 - self.n >= 0:
            yield swap(tile0-self.n)
        if tile0 +self.n < self.n2:
            yield swap(tile0+self.n)
        if tile0 % self.n > 0:
            yield swap(tile0-1)
        if tile0 % self.n < self.n-1:
            yield swap(tile0+1)



def test_block(num_tests = 1):
    x_m = [None]*num_tests
    y_m = [None]*num_tests
    x_d = [None]*num_tests
    y_d = [None]*num_tests
    x_g = [None]*num_tests
    y_g = [None]*num_tests

    for k in xrange(num_tests):
        p = shuffle(PuzzleBlock(3), 200)
        print "Puzzle Number: ", (k+1)
        print p.showMatrix()
        print "Misplaced Heuristic: ",
        x_m[k],y_m[k] = solve(p, PuzzleBlock(3), misplaced_heuristic)
        print "Distance Heuristic: ",
        x_d[k],y_d[k] = solve(p, PuzzleBlock(3), distance_heuristic)
        print "Gaschnig Heuristic: ",
        x_g[k],y_g[k] = solve(p, PuzzleBlock(3), gaschnig_heuristic)
        print "================================================================"

    x_m_average = sum(x_m) / num_tests
    y_m_average = sum(y_m) / num_tests
    x_d_average = sum(x_d) / num_tests
    y_d_average = sum(y_d) / num_tests
    x_g_average = sum(x_g) / num_tests
    y_g_average = sum(y_g) / num_tests


    print "Average number of steps for Misplaced Tiles heuristic: " + str(x_m_average)
    print "Average number of nodes expanded for Misplaced Tiles heuristic: " + str(y_m_average)
    print "Average number of steps for Manhattan distance heuristic: " + str(x_d_average)
    print "Average number of nodes expanded for Manhattan distance heuristic: " + str(y_d_average)
    print "Average number of steps for Gaschnig heuristic: " + str(x_g_average)
    print "Average number of nodes expanded for Gaschnig heuristic: " + str(y_g_average)

# Generate the scatterplot
    pl1 = matplotlib.pyplot.scatter(x_m,y_m,color='black',marker='^')
    pl2 = matplotlib.pyplot.scatter(x_d,y_d,color='red',marker='*')
    pl3 = matplotlib.pyplot.scatter(x_g,y_g,color='green')

    matplotlib.pyplot.legend((pl1,pl2,pl3),('Misplaced Tiles','Manhattan Distance','Gaschnig'),
    scatterpoints=1,
    ncol=3,
    fontsize=10)

    matplotlib.pyplot.xlabel('Steps')
    matplotlib.pyplot.ylabel('Nodes Expanded')
    # matplotlib.pyplot.plot(label="demo")
    matplotlib.pyplot.show()


def main():
# Pass number of instances you want to generate
    instances = int(raw_input("Enter number of puzzle instances to generate: "))
    test_block(instances)

if __name__ == "__main__":
    main()

from board_rep import *

# MAIN EVALUATIONS

def main_evaluation(pos):
    mg = middle_game_evaluation(pos)
    eg = end_game_evaluation(pos)
    p = phase(pos)
    rule50_val = rule50(pos)
    eg = eg * scale_factor(pos, eg) / 64
    v = (((mg * p + ((eg * (128 - p)) // 1)) // 128) // 1)
    if pos is not None:
        v = ((v // 16) // 1) * 16
    v += tempo(pos)
    v = (v * (100 - rule50_val) // 100) // 1
    return v

def middle_game_evaluation(pos, nowinnable = True):
    v = 0
    v += piece_value_mg(pos) - piece_value_mg(colorflip(pos))
    v += psqt_mg(pos) - psqt_mg(colorflip(pos))
    v += imbalance_total(pos)
    v += pawns_mg(pos) - pawns_mg(colorflip(pos))
    v += pieces_mg(pos) - pieces_mg(colorflip(pos))
    v += mobility_mg(pos) - mobility_mg(colorflip(pos))
    v += threats_mg(pos) - threats_mg(colorflip(pos))
    v += passed_mg(pos) - passed_mg(colorflip(pos))
    v += space(pos) - space(colorflip(pos))
    v += king_mg(pos) - king_mg(colorflip(pos))
    if not nowinnable:
        v += winnable_total_mg(pos, v)
    return v

def end_game_evaluation(pos, nowinnable = True):
    v = 0
    v += piece_value_eg(pos) - piece_value_eg(colorflip(pos))
    v += psqt_eg(pos) - psqt_eg(colorflip(pos))
    v += imbalance_total(pos)
    v += pawns_eg(pos) - pawns_eg(colorflip(pos))
    v += pieces_eg(pos) - pieces_eg(colorflip(pos))
    v += mobility_eg(pos) - mobility_eg(colorflip(pos))
    v += threats_eg(pos) - threats_eg(colorflip(pos))
    v += passed_eg(pos) - passed_eg(colorflip(pos))
    v += king_eg(pos) - king_eg(colorflip(pos))
    if not nowinnable:
        v += winnable_total_eg(pos, v)
    return v


# EVALUATION FUNCTIONS -----------

# BEGIN ATTACK

def pinned_direction(pos, square = None):
    if square is None:
        return sum(pos, pinned_direction)
    if board(pos, square['x'], square['y']).upper() not in "PNBRQK":
        return 0
    color = 1
    if board(pos, square['x'], square['y']) not in "PNBRQK":
        color = -1
    for i in range(8):
        ix = (i + (i > 3)) % 3 - 1
        iy = (((i + (i > 3)) // 3) << 0) - 1
        king = False
        for d in range(1, 8):
            b = board(pos, square['x'] + d * ix, square['y'] + d * iy)
            if b == "K":
                king = True
            if b != "-":
                break
        if king:
            for d in range(1, 8):
                b = board(pos, square['x'] - d * ix, square['y'] - d * iy)
                if b == "q" or (b == "b" and ix * iy != 0) or (b == "r" and ix * iy == 0):
                    return abs(ix + iy * 3) * color
                if b != "-":
                    break
    return 0

def knight_attack(pos, square = None, s2 = None):
    if square is None:
        return sum(pos, knight_attack)
    v = 0
    for i in range(8):
        ix = ((i > 3) + 1) * (((i % 4) > 1) * 2 - 1)
        iy = (2 - (i > 3)) * ((i % 2 == 0) * 2 - 1)
        b = board(pos, square['x'] + ix, square['y'] + iy)
        if b == "N" and (s2 is None or s2['x'] == square['x'] + ix and s2['y'] == square['y'] + iy) and not pinned(pos, {"x": square['x'] + ix, "y": square['y'] + iy}):
            v += 1
    return v

def bishop_xray_attack(pos, square = None, s2 = None):
    if square is None:
        return sum(pos, bishop_xray_attack)
    v = 0
    for i in range(4):
        ix = ((i > 1) * 2 - 1)
        iy = ((i % 2 == 0) * 2 - 1)
        for d in range(1, 8):
            b = board(pos, square['x'] + d * ix, square['y'] + d * iy)
            if b == "B" and (s2 is None or (s2['x'] == square['x'] + d * ix and s2['y'] == square['y'] + d * iy)):
                pinned_dir = pinned_direction(pos, {"x": square['x'] + d * ix, "y": square['y'] + d * iy})
                if pinned_dir == 0 or abs(ix + iy * 3) == pinned_dir:
                    v += 1
            if b != "-" and b != "Q" and b != "q":
                break
    return v

def rook_xray_attack(pos, square = None, s2 = None):
    if square is None:
        return sum(pos, rook_xray_attack)
    v = 0
    for i in range(4):
        ix = -1 if i == 0 else 1 if i == 1 else 0
        iy = -1 if i == 2 else 1 if i == 3 else 0
        for d in range(1, 8):
            b = board(pos, int(square['x'] + d * ix), int(square['y'] + d * iy))
            if b == "R" and (s2 is None or (s2['x'] == square['x'] + d * ix and s2['y'] == square['y'] + d * iy)):
                pinned_dir = pinned_direction(pos, {"x": square['x'] + d * ix, "y": square['y'] + d * iy})
                if pinned_dir == 0 or abs(ix + iy * 3) == pinned_dir:
                    v += 1
            if b != "-" and b != "R" and b != "Q" and b != "q":
                break
    return v

def queen_attack(pos, square = None, s2 = None):
    if square is None:
        return sum(pos, queen_attack)
    v = 0
    for i in range(8):
        ix = (i + (i > 3)) % 3 - 1
        iy = (((i + (i > 3)) // 3) << 0) - 1
        for d in range(1, 8):
            b = board(pos, int(square['x'] + d * ix), int(square['y'] + d * iy))
            if b == "Q" and (s2 is None or (s2['x'] == square['x'] + d * ix and s2['y'] == square['y'] + d * iy)):
                pinned_dir = pinned_direction(pos, {"x": square['x'] + d * ix, "y": square['y'] + d * iy})
                if pinned_dir == 0 or abs(ix + iy * 3) == pinned_dir:
                    v += 1
            if b != "-":
                break
    return v


def pawn_attack(pos, square = None):
    if square is None:
        return sum(pos, pawn_attack)
    v = 0
    if board(pos, square['x'] - 1, square['y'] + 1) == "P":
        v += 1
    if board(pos, square['x'] + 1, square['y'] + 1) == "P":
        v += 1
    return v

def king_attack(pos, square = None):
    if square is None:
        return sum(pos, king_attack)
    for i in range(8):
        ix = (i + (i > 3)) % 3 - 1
        iy = (((i + (i > 3)) // 3) << 0) - 1
        if board(pos, square['x'] + ix, square['y'] + iy) == "K":
            return 1
    return 0

def attack(pos, square = None):
    if square is None:
        return sum(pos, attack)
    v = 0
    v += pawn_attack(pos, square = square)
    v += king_attack(pos, square = square)
    v += knight_attack(pos, s2 = None, square = square)
    v += bishop_xray_attack(pos, s2 = None, square = square)
    v += rook_xray_attack(pos, s2 = None, square = square)
    v += queen_attack(pos, s2 = None, square = square)
    return v

def queen_attack_diagonal(pos, s2 = None, square = None):
    if square is None:
        return sum(pos, queen_attack_diagonal)
    v = 0
    for i in range(8):
        ix = (i + (i > 3)) % 3 - 1
        iy = (((i + (i > 3)) // 3) << 0) - 1
        if ix == 0 or iy == 0:
            continue
        for d in range(1, 8):
            b = board(pos, square['x'] + d * ix, square['y'] + d * iy)
            if b == "Q" and (s2 is None or s2['x'] == square['x'] + d * ix and s2['y'] == square['y'] + d * iy):
                dir = pinned_direction(pos, {"x": square['x'] + d * ix, "y": square['y'] + d * iy})
                if dir == 0 or abs(ix + iy * 3) == dir:
                    v += 1
            if b != "-":
                break
    return v

def pinned(pos, square = None):
    if square is None:
        return sum(pos, pinned)
    if board(pos, square['x'], square['y']) not in "PNBRQK":
        return 0
    return 1 if pinned_direction(pos, square) > 0 else 0

# END ATTACK

# BEGIN HELPERS

def rank(pos, square = None):
    if square is None:
        return sum(pos, rank)
    return 8 - square['y']

def ffile(pos, square = None):
    if square is None:
        return sum(pos, ffile)
    return 1 + square['x']

def bishop_count(pos, square = None):
    if square is None:
        return sum(pos, bishop_count)
    if board(pos, square['x'], square['y']) == "B":
        return 1
    return 0

def queen_count(pos, square = None):
    if square is None:
        return sum(pos, queen_count)
    if board(pos, square['x'], square['y']) == "Q":
        return 1
    return 0

def pawn_count(pos, square = None):
    if square is None:
        return sum(pos, pawn_count)
    if board(pos, square['x'], square['y']) == "P":
        return 1
    return 0

def knight_count(pos, square = None):
    if square is None:
        return sum(pos, knight_count)
    if board(pos, square['x'], square['y']) == "N":
        return 1
    return 0

def rook_count(pos, square = None):
    if square is None:
        return sum(pos, rook_count)
    if board(pos, square['x'], square['y']) == "R":
        return 1
    return 0

def opposite_bishops(pos):
    if bishop_count(pos) != 1:
        return 0
    if bishop_count(colorflip(pos)) != 1:
        return 0
    color = [0, 0]
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "B":
                color[0] = (x + y) % 2
            if board(pos, x, y) == "b":
                color[1] = (x + y) % 2
    return 0 if color[0] == color[1] else 1

def king_distance(pos, square = None):
    if square is None:
        return sum(pos, king_distance)
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "K":
                return max(abs(x - square['x']), abs(y - square['y']))
    return 0

def king_ring(pos, square, full):
    if square is None:
        return sum(pos, king_ring)
    if not full and board(pos, square['x'] + 1, square['y'] - 1) == "p" and board(pos, square['x'] - 1, square['y'] - 1) == "p":
        return 0
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            if board(pos, square['x'] + ix, square['y'] + iy) == "k" and (ix >= -1 and ix <= 1 or square['x'] + ix == 0 or square['x'] + ix == 7) and (iy >= -1 and iy <= 1 or square['y'] + iy == 0 or square['y'] + iy == 7):
                return 1
    return 0

def piece_count(pos, square = None):
    if square is None:
        return sum(pos, piece_count)

    if board(pos, square['x'], square['y']) not in "PNBRQK":
        return 0

    i = "PNBRQK".index(board(pos, square['x'], square['y']))
    return 1 if i >= 0 else 0

def pawn_attacks_span(pos, square = None):
    if square is None:
        return sum(pos, pawn_attacks_span)
    pos2 = colorflip(pos)
    for y in range(square['y']):
        if board(pos, square['x'] - 1, y) == "p" and (y == square['y'] - 1 or (board(pos, square['x'] - 1, y + 1) != "P" and not backward(pos2, {'x':square['x']-1, 'y':7-y}))):
            return 1
        if board(pos, square['x'] + 1, y) == "p" and (y == square['y'] - 1 or (board(pos, square['x'] + 1, y + 1) != "P" and not backward(pos2, {'x':square['x']+1, 'y':7-y}))):
            return 1
    return 0


# END HELPERS

# BEGIN IMBALANCE

def imbalance(pos, square = None):
    if square is None:
        return sum(pos, imbalance)
    qo = [[0],[40,38],[32,255,-62],[0,104,4,0],[-26,-2,47,105,-208],[-189,24,117,133,-134,-6]]
    qt = [[0],[36,0],[9,63,0],[59,65,42,0],[46,39,24,-24,0],[97,100,-42,137,268,0]]

    if (board(pos, square['x'], square['y']) == '-' or 'K' or 'k') or board(pos, square['x'], square['y']) not in "XPNBRQxpnbrq":
        return 0

    j = "XPNBRQxpnbrq".index(board(pos, square['x'], square['y']))
    if j < 0 or j > 5:
        return 0
    bishop = [0, 0]
    v = 0
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) not in "XPNBRQxpnbrq":
                continue

            i = "XPNBRQxpnbrq".index(board(pos, x, y))
            if i < 0:
                continue
            if i == 9:
                bishop[0] += 1
            if i == 3:
                bishop[1] += 1
            if i % 6 > j:
                continue
            if i > 5:
                v += qt[j][i-6]
            else:
                v += qo[j][i]
    if bishop[0] > 1:
        v += qt[j][0]
    if bishop[1] > 1:
        v += qo[j][0]
    return v

def bishop_pair(pos, square = None):
    if bishop_count(pos) < 2:
        return 0
    if square is None:
        return 1438
    return 1 if board(pos, square['x'], square['y']) == "B" else 0

def imbalance_total(pos, square = None):
    v = 0
    v += imbalance(pos) - imbalance(colorflip(pos))
    v += bishop_pair(pos) - bishop_pair(colorflip(pos))
    return int(v / 16)


# END IMBALANCE

# BEGIN KING

def pawnless_flank(pos):
    pawns=[0,0,0,0,0,0,0,0]
    kx = 0
    for x in range(8):
        for y in range(8):
            if board(pos, x, y).upper() == "P":
                pawns[x] += 1
            if board(pos, x, y) == "k":
                kx = x
    if kx == 0:
        sum = pawns[0] + pawns[1] + pawns[2]
    elif kx < 3:
        sum = pawns[0] + pawns[1] + pawns[2] + pawns[3]
    elif kx < 5:
        sum = pawns[2] + pawns[3] + pawns[4] + pawns[5]
    elif kx < 7:
        sum = pawns[4] + pawns[5] + pawns[6] + pawns[7]
    else:
        sum = pawns[5] + pawns[6] + pawns[7]
    return 1 if sum == 0 else 0

def strength_square(pos, square = None):
    if square == None:
        return sum(pos, strength_square)
    v = 5
    kx = min(6, max(1, square['x']))
    weakness = [[-6,81,93,58,39,18,25],
                [-43,61,35,-49,-29,-11,-63],
                [-10,75,23,-2,32,3,-45],
                [-39,-13,-29,-52,-48,-67,-166]]
    for x in range(kx - 1, kx + 2):
        us = 0
        for y in range(7, square['y'] - 1, -1):
            if board(pos, x, y) == "p" and board(pos, x-1, y+1) != "P" and board(pos, x+1, y+1) != "P":
                us = y
        f = min(x, 7 - x)
        v += weakness[f][us] if weakness[f][us] else 0
    return v

def storm_square(pos, square = None, eg = False):
    if square == None:
        return sum(pos, storm_square)
    v = 0
    ev = 5
    kx = min(6, max(1, square['x']))
    unblockedstorm = [[85,-289,-166,97,50,45,50],
                      [46,-25,122,45,37,-10,20],
                      [-6,51,168,34,-2,-22,-14],
                      [-15,-11,101,4,11,-15,-29]]
    blockedstorm = [[0,0,76,-10,-7,-4,-1],
                    [0,0,78,15,10,6,2]]
    for x in range(kx - 1, kx + 2):
        us = 0
        them = 0
        for y in range(7, square['y'] - 1, -1):
            if board(pos, x, y) == "p" and board(pos, x-1, y+1) != "P" and board(pos, x+1, y+1) != "P":
                us = y
            if board(pos, x, y) == "P":
                them = y
        f = min(x, 7 - x)
        if us > 0 and them == us + 1:
            v += blockedstorm[0][them]
            ev += blockedstorm[1][them]
        else:
            v += unblockedstorm[f][them]
    return ev if eg else v

def shelter_strength(pos, square = None):
    w = 0
    s = 1024
    tx = None
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "k" or (pos['c'][2] and x == 6 and y == 0) or (pos['c'][3] and x == 2 and y == 0):
                w1 = strength_square(pos, square = {"x":x,"y":y})
                s1 = storm_square(pos, square = {"x":x,"y":y})
                if s1 - w1 < s - w:
                    w = w1
                    s = s1
                    tx = max(1, min(6, x))
    if square == None:
        return w
    if tx != None and board(pos, square['x'], square['y']) == "p" and square['x'] >= tx-1 and square['x'] <= tx+1:
        for y in range(square['y']-1, -1, -1):
            if board(pos, square['x'], y) == "p":
                return 0
        return 1
    return 0

def shelter_storm(pos, square = None):
    w = 0
    s = 1024
    tx = None
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "k" or (pos['c'][2] and x == 6 and y == 0) or (pos['c'][3] and x == 2 and y == 0):
                w1 = strength_square(pos, square = {"x": x, "y": y})
                s1 = storm_square(pos, square = {"x": x, "y": y})
                if s1 - w1 < s - w:
                    w = w1
                    s = s1
                    tx = max(1, min(6, x))
    if square is None:
        return s
    if tx is not None and board(pos, square["x"], square["y"]).upper() == "P" and square["x"] >= tx - 1 and square["x"] <= tx + 1:
        for y in range(square["y"] - 1, -1, -1):
            if board(pos, square["x"], y) == board(pos, square["x"], square["y"]):
                return 0
        return 1
    return 0

def king_pawn_distance(pos, square = None):
    v = 6
    kx = 0
    ky = 0
    px = 0
    py = 0
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "K":
                kx = x
                ky = y
    for x in range(8):
        for y in range(8):
            dist = max(abs(x - kx), abs(y - ky))
            if board(pos, x, y) == "P" and dist < v:
                px = x
                py = y
                v = dist
    if square is None or (square["x"] == px and square["y"] == py):
        return v
    return 0

def check(pos, square = None, type_piece = None):
    if square is None:
        return sum(pos, check)
    if (rook_xray_attack(pos, s2 = None, square = square) and (type_piece is None or type_piece == 2 or type_piece == 4)) or (queen_attack(pos, s2 = None, square = square) and (type_piece is None or type_piece == 3)):
        for i in range(4):
            ix = -1 if i == 0 else 1 if i == 1 else 0
            iy = -1 if i == 2 else 1 if i == 3 else 0
            for d in range(1, 8):
                b = board(pos, square["x"] + d * ix, square["y"] + d * iy)
                if b == "k":
                    return 1
                if b != "-" and b != "q":
                    break
    if (bishop_xray_attack(pos, s2 = None, square = square) and (type_piece is None or type_piece == 1 or type_piece == 4)) or (queen_attack(pos, s2 = None, square = square) and (type_piece is None or type_piece == 3)):
        for i in range(4):
            ix = (2 * (i > 1) - 1)
            iy = (2 * (i % 2 == 0) - 1)
            for d in range(1, 8):
                b = board(pos, square["x"] + d * ix, square["y"] + d * iy)
                if b == "k":
                    return 1
                if b != "-" and b != "q":
                    break
    if knight_attack(pos, square = square) and (type_piece is None or type_piece == 0 or type_piece == 4):
        if (board(pos, square["x"] + 2, square["y"] + 1) == "k" or
            board(pos, square["x"] + 2, square["y"] - 1) == "k" or
            board(pos, square["x"] + 1, square["y"] + 2) == "k" or
            board(pos, square["x"] + 1, square["y"] - 2) == "k" or
            board(pos, square["x"] - 2, square["y"] + 1) == "k" or
            board(pos, square["x"] - 2, square["y"] - 1) == "k" or
            board(pos, square["x"] - 1, square["y"] + 2) == "k" or
            board(pos, square["x"] - 1, square["y"] - 2) == "k"):
            return 1
    return 0

def safe_check(pos, type_piece = None, square = None):
    #print(square)
    if square is None:
        return sum(pos, safe_check, type_piece)
    if board(pos, square['x'], square['y']) is None or board(pos, square['x'], square['y']) not in ['P', 'N', 'B', 'R', 'Q', 'K']:
        return 0
    if "PNBRQK".index(board(pos, square['x'], square['y'])) >= 0:
        return 0
    if not check(pos, square, type_piece):
        return 0
    pos2 = colorflip(pos)
    if type_piece == 3 and safe_check(pos, type_piece = 2, square = square):
        return 0
    if type_piece == 1 and safe_check(pos, type_piece = 3, square = square):
        return 0
    if ((not attack(pos2, {"x": square['x'], "y": 7 - square['y']}) 
        or (weak_squares(pos, square) and attack(pos, square) > 1))
        and (type_piece != 3 or not queen_attack(pos2, {"x": square['x'], "y": 7 - square['y']}))):
        return 1
    return 0

def king_attackers_count(pos, square = None):
    if square is None:
        return sum(pos, king_attackers_count)
    if board(pos, square['x'], square['y']) is None or board(pos, square['x'], square['y']) not in "PNBRQ":
        return 0
    if board(pos, square['x'], square['y']) == "P":
        v = 0
        for dir in range(-1, 2, 2):
            fr = board(pos, square['x'] + dir * 2, square['y']) == "P"
            if square['x'] + dir >= 0 and square['x'] + dir <= 7 and king_ring(pos, {"x": square['x'] + dir, "y": square['y'] - 1}, True):
                v = v + (0.5 if fr else 1)
        return v
    for x in range(8):
        for y in range(8):
            s2 = {"x": x, "y": y}
            if king_ring(pos, s2, full = None):
                if (knight_attack(pos, s2 = s2, square = square)
                    or bishop_xray_attack(pos, s2 = s2, square = square)
                    or rook_xray_attack(pos, s2 = s2, square = square)
                    or queen_attack(pos, s2 = s2, square = square)):
                    return 1
    return 0

def king_attackers_weight(pos, square = None):
    if square is None:
        return sum(pos, king_attackers_weight)
    if king_attackers_count(pos, square):
        #print(board(pos, square['x'], square['y']))
        #print(king_attackers_count(pos, square))
        #print(pos)
        #print(square)
        return [0, 81, 52, 44, 10]["PNBRQ".index(board(pos, square['x'], square['y']))]
    return 0

def king_attacks(pos, square = None):
    if square is None:
        return sum(pos, king_attacks)
    if board(pos, square['x'], square['y']) is None or board(pos, square['x'], square['y']) not in ['N', 'B', 'R', 'Q']:
        return 0
    if "NBRQ".index(board(pos, square['x'], square['y'])) < 0:
        return 0
    if king_attackers_count(pos, square) == 0:
        return 0
    kx = 0
    ky = 0
    v = 0
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "k":
                kx = x
                ky = y
    for x in range(kx - 1, kx + 2):
        for y in range(ky - 1, ky + 2):
            s2 = {"x": x, "y": y}
            if x >= 0 and y >= 0 and x <= 7 and y <= 7 and (x != kx or y != ky):
                v += knight_attack(pos, s2 = s2, square = square)
                v += bishop_xray_attack(pos, s2 = s2, square = square)
                v += rook_xray_attack(pos, s2 = s2, square = square)
                v += queen_attack(pos, s2 = s2, square = square)
    return v

def weak_bonus(pos, square = None):
    if square is None:
        return sum(pos, weak_bonus)
    if not weak_squares(pos, square):
        return 0
    if not king_ring(pos, square, None):
        return 0
    return 1

def weak_squares(pos, square = None):
    if square is None:
        return sum(pos, weak_squares)
    if attack(pos, square):
        pos2 = colorflip(pos)
        attack_val = attack(pos2, {'x': square['x'], 'y': 7 - square['y']})
        if attack_val >= 2:
            return 0
        if attack_val == 0:
            return 1
        if king_attack(pos2, {'x': square['x'], 'y': 7 - square['y']}) or queen_attack(pos2, {'x': square['x'], 'y': 7 - square['y']}):
            return 1
    return 0

def unsafe_checks(pos, square = None):
    if square is None:
        return sum(pos, unsafe_checks)
    if check(pos, square, 0) and safe_check(pos, square = None, type_piece = 0) == 0:
        return 1
    if check(pos, square, 1) and safe_check(pos, square = None, type_piece = 1) == 0:
        return 1
    if check(pos, square, 2) and safe_check(pos, square = None, type_piece = 2) == 0:
        return 1
    return 0

def knight_defender(pos, square = None):
    if square is None:
        return sum(pos, knight_defender)
    if knight_attack(pos, s2 = None, square = square) and king_attack(pos, square):
        return 1
    return 0

def endgame_shelter(pos, square = None):
    w = 0
    s = 1024
    tx = None
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "k" or (pos['c'][2] and x == 6 and y == 0) or (pos['c'][3] and x == 2 and y == 0):
                w1 = strength_square(pos, square = {'x': x, 'y': y})
                s1 = storm_square(pos, square = {'x': x, 'y': y})
                e1 = storm_square(pos, square = {'x': x, 'y': y}, eg = True)
                if s1 - w1 < s - w:
                    w = w1
                    s = s1
                    e = e1
    if square is None:
        return e
    return 0

def blockers_for_king(pos, square = None):
    if square is None:
        return sum(pos, blockers_for_king)
    if pinned_direction(colorflip(pos), {'x': square['x'], 'y': 7 - square['y']}):
        return 1
    return 0

def flank_attack(pos, square = None):
    if square is None:
        return sum(pos, flank_attack)
    if square['y'] > 4:
        return 0
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "k":
                if x == 0 and square['x'] > 2:
                    return 0
                if x < 3 and square['x'] > 3:
                    return 0
                if x >= 3 and x < 5 and (square['x'] < 2 or square['x'] > 5):
                    return 0
                if x >= 5 and square['x'] < 4:
                    return 0
                if x == 7 and square['x'] < 5:
                    return 0
    a = attack(pos, square)
    if not a:
        return 0
    return 2 if a > 1 else 1

def flank_defense(pos, square = None):
    if square is None:
        return sum(pos, flank_defense)
    if square['y'] > 4:
        return 0
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "k":
                if x == 0 and square['x'] > 2:
                    return 0
                if x < 3 and square['x'] > 3:
                    return 0
                if x >= 3 and x < 5 and (square['x'] < 2 or square['x'] > 5):
                    return 0
                if x >= 5 and square['x'] < 4:
                    return 0
                if x == 7 and square['x'] < 5:
                    return 0
    return 1 if attack(colorflip(pos), {'x': square['x'], 'y': 7 - square['y']}) > 0 else 0

def king_danger(pos):
    count = king_attackers_count(pos)
    weight = king_attackers_weight(pos)
    kingAttacks = king_attacks(pos)
    weak = weak_bonus(pos)
    unsafeChecks = unsafe_checks(pos)
    blockersForKing = blockers_for_king(pos)
    kingFlankAttack = flank_attack(pos)
    kingFlankDefense = flank_defense(pos)
    noQueen = 0 if (queen_count(pos) > 0) else 1
    v = count * weight + 69 * kingAttacks + 185 * weak - 100 * (knight_defender(colorflip(pos)) > 0) + 148 * unsafeChecks + 98 * blockersForKing - 4 * kingFlankDefense + (int((3 * kingFlankAttack * kingFlankAttack / 8)) << 0) - 873 * noQueen - (int((6 * (shelter_strength(pos) - shelter_storm(pos)) / 8)) << 0) + mobility_mg(pos) - mobility_mg(colorflip(pos)) + 37 + (int((772 * min(safe_check(pos, type_piece = 3, square = None), 1.45))) << 0) + (int(1084 * min(safe_check(pos, type_piece = 2, square = None), 1.75)) << 0) + (int(645 * min(safe_check(pos, type_piece = 1, square = None), 1.50)) << 0) + (int(792 * min(safe_check(pos, type_piece = 0, square = None), 1.62)) << 0)
    if v > 100:
        return v
    return 0

def king_mg(pos):
    v = 0
    kd = king_danger(pos)
    v -= shelter_strength(pos)
    v += shelter_storm(pos)
    v += int(kd * kd / 4096) << 0
    v += 8 * flank_attack(pos)
    v += 17 * pawnless_flank(pos)
    return v

def king_eg(pos):
    v = 0
    v -= 16 * king_pawn_distance(pos)
    v += endgame_shelter(pos)
    v += 95 * pawnless_flank(pos)
    v += int(king_danger(pos) / 16) << 0
    return v

# END KING

# BEGIN MATERIAL

def non_pawn_material(pos, square = None):
    if square is None:
        return sum(pos, non_pawn_material)
    if (board(pos, square['x'], square['y']) == '-' or 'K' or 'P') or board(pos, square['x'], square['y']).islower():
        return 0
    i = "NBRQ".index(board(pos, square['x'], square['y']))
    if i >= 0:
        return piece_value_bonus(pos, True, square)
    return 0

def piece_value_bonus(pos, mg, square = None):
    if square is None:
        return sum(pos, piece_value_bonus)
    a = [124, 781, 825, 1276, 2538] if mg else [206, 854, 915, 1380, 2682]
    
    if (board(pos, square['x'], square['y']) == '-' or 'k') or board(pos, square['x'], square['y']).islower():
        return 0

    i = "PNBRQ".index(board(pos, square['x'], square['y']))
    if i >= 0:
        return a[i]
    return 0

def psqt_bonus(pos, mg, square = None):
    if square is None:
        return sum(pos, psqt_bonus, mg)
    bonus = [
        [[-175,-92,-74,-73],[-77,-41,-27,-15],[-61,-17,6,12],[-35,8,40,49],[-34,13,44,51],[-9,22,58,53],[-67,-27,4,37],[-201,-83,-56,-26]],
        [[-53,-5,-8,-23],[-15,8,19,4],[-7,21,-5,17],[-5,11,25,39],[-12,29,22,31],[-16,6,1,11],[-17,-14,5,0],[-48,1,-14,-23]],
        [[-31,-20,-14,-5],[-21,-13,-8,6],[-25,-11,-1,3],[-13,-5,-4,-6],[-27,-15,-4,3],[-22,-2,6,12],[-2,12,16,18],[-17,-19,-1,9]],
        [[3,-5,-5,4],[-3,5,8,12],[-3,6,13,7],[4,5,9,8],[0,14,12,5],[-4,10,6,8],[-5,6,10,8],[-2,-2,1,-2]],
        [[271,327,271,198],[278,303,234,179],[195,258,169,120],[164,190,138,98],[154,179,105,70],[123,145,81,31],[88,120,65,33],[59,89,45,-1]]
    ] if mg else [
        [[-96,-65,-49,-21],[-67,-54,-18,8],[-40,-27,-8,29],[-35,-2,13,28],[-45,-16,9,39],[-51,-44,-16,17],[-69,-50,-51,12],[-100,-88,-56,-17]],
        [[-57,-30,-37,-12],[-37,-13,-17,1],[-16,-1,-2,10],[-20,-6,0,17],[-17,-1,-14,15],[-30,6,4,6],[-31,-20,-1,1],[-46,-42,-37,-24]],
        [[-9,-13,-10,-9],[-12,-9,-1,-2],[6,-8,-2,-6],[-6,1,-9,7],[-5,8,7,-6],[6,1,-7,10],[4,5,20,-5],[18,0,19,13]],
        [[-69,-57,-47,-26],[-55,-31,-22,-4],[-39,-18,-9,3],[-23,-3,13,24],[-29,-6,9,21],[-38,-18,-12,1],[-50,-27,-24,-8],[-75,-52,-43,-36]],
        [[1,45,85,76],[53,100,133,135],[88,130,169,175],[103,156,172,172],[96,166,199,199],[92,172,184,191],[47,121,116,131],[11,59,73,78]]
    ]
    pbonus = [
        [0,0,0,0,0,0,0,0],[3,3,10,19,16,19,7,-5],[-9,-15,11,15,32,22,5,-22],[-4,-23,6,20,40,17,4,-8],[13,0,-13,1,11,-2,-13,5],
        [5,-12,-7,22,-8,-5,-15,-8],[-7,7,-3,-13,5,-16,10,-8],[0,0,0,0,0,0,0,0]
    ] if mg else [
        [0,0,0,0,0,0,0,0],[-10,-6,10,0,14,7,-5,-19],[-10,-10,-10,4,4,3,-6,-4],[6,-2,-8,-4,-13,-12,-10,-9],[10,5,4,-5,-5,-5,14,9],
        [28,20,21,28,30,7,6,13],[0,-11,12,21,25,19,4,7],[0,0,0,0,0,0,0,0]
    ]
    if board(pos, square['x'], square['y']) == '-' or board(pos, square['x'], square['y']).islower():
        return 0

    i = "PNBRQK".index(board(pos, square['x'], square['y']))
    if i < 0:
        return 0
    if i == 0:
        return pbonus[7 - square['y']][square['x']]
    else:
        return bonus[i-1][7 - square['y']][min(square['x'], 7 - square['x'])]

def piece_value_mg(pos, square = None):
    if square is None:
        return sum(pos, piece_value_mg)
    return piece_value_bonus(pos, True, square)

def piece_value_eg(pos, square = None):
    if square is None:
        return sum(pos, piece_value_eg)
    return piece_value_bonus(pos, False, square)

def psqt_mg(pos, square = None):
    if square is None:
        return sum(pos, psqt_mg)
    return psqt_bonus(pos, True, square)

def psqt_eg(pos, square = None):
    if square is None:
        return sum(pos, psqt_eg)
    return psqt_bonus(pos, False, square)


# END MATERIAL

# BEGIN MOBILITY

def mobility(pos, square = None):
    if square is None:
        return sum(pos, mobility)
    v = 0
    b = board(pos, square['x'], square['y'])
    if b not in "NBRQ":
        return 0
    for x in range(8):
        for y in range(8):
            s2 = {"x": x, "y": y}
            if not mobility_area(pos, s2):
                continue
            if b == "N" and knight_attack(pos, s2 = s2, square = square) and board(pos, x, y) != 'Q':
                v += 1
            if b == "B" and bishop_xray_attack(pos, s2 = s2, square = square) and board(pos, x, y) != 'Q':
                v += 1
            if b == "R" and rook_xray_attack(pos, s2 = s2, square = square):
                v += 1
            if b == "Q" and queen_attack(pos, s2 = s2, square = square):
                v += 1
    return v

def mobility_area(pos, square = None):
    if square is None:
        return sum(pos, mobility_area)
    if board(pos, square['x'], square['y']) == "K":
        return 0
    if board(pos, square['x'], square['y']) == "Q":
        return 0
    if board(pos, square['x'] - 1, square['y'] - 1) == "p":
        return 0
    if board(pos, square['x'] + 1, square['y'] - 1) == "p":
        return 0
    if board(pos, square['x'], square['y']) == "P" and (rank(pos, square) < 4 or board(pos, square['x'], square['y'] - 1) != "-"):
        return 0
    if blockers_for_king(colorflip(pos), {"x": square['x'], "y": 7 - square['y']}):
        return 0
    return 1

def mobility_bonus(pos, square, mg):
    if square is None:
        return sum(pos, mobility_bonus, mg)
    bonus = [[-62,-53,-12,-4,3,13,22,28,33],
             [-48,-20,16,26,38,51,55,63,63,68,81,81,91,98],
             [-60,-20,2,3,3,11,22,31,40,40,41,48,57,57,62],
             [-30,-12,-8,-9,20,23,23,35,38,53,64,65,65,66,67,67,72,72,77,79,93,108,108,108,110,114,114,116]] if mg else [[-81,-56,-31,-16,5,11,17,20,25],
                                                                                                                                 [-59,-23,-3,13,24,42,54,57,65,73,78,86,88,97],
                                                                                                                                 [-78,-17,23,39,70,99,103,121,134,139,158,164,168,169,172],
                                                                                                                                 [-48,-30,-7,19,40,55,59,75,78,96,96,100,121,127,131,133,136,141,147,150,151,168,168,171,182,182,192,219]]
    if board(pos, square['x'], square['y']) not in ['N', 'B', 'R', 'Q']:
        return 0

    i = "NBRQ".index(board(pos, square['x'], square['y']))
    if i < 0:
        return 0
    return bonus[i][mobility(pos, square)]

def mobility_mg(pos, square = None):
    if square is None:
        return sum(pos, mobility_mg)
    return mobility_bonus(pos, square, True)

def mobility_eg(pos, square = None):
    if square is None:
        return sum(pos, mobility_eg)
    return mobility_bonus(pos, square, False)

# END MOBILITY

# BEGIN PASSED PAWNS

def candidate_passed(pos, square = None):
    if square is None:
        return sum(pos, candidate_passed)
    if board(pos, square['x'], square['y']) != "P":
        return 0
    ty1 = 8
    ty2 = 8
    oy = 8
    for y in range(square['y'] - 1, -1, -1):
        if board(pos, square['x'], y) == "P":
            return 0
        if board(pos, square['x'], y) == "p":
            ty1 = y
        if board(pos, square['x'] - 1, y) == "p" or board(pos, square['x'] + 1, y) == "p":
            ty2 = y
    if ty1 == 8 and ty2 >= square['y'] - 1:
        return 1
    if ty2 < square['y'] - 2 or ty1 < square['y'] - 1:
        return 0
    if ty2 >= square['y'] and ty1 == square['y'] - 1 and square['y'] < 4:
        if board(pos, square['x'] - 1, square['y'] + 1) == "P" and board(pos, square['x'] - 1, square['y']) != "p" and board(pos, square['x'] - 2, square['y'] - 1) != "p":
            return 1
        if board(pos, square['x'] + 1, square['y'] + 1) == "P" and board(pos, square['x'] + 1, square['y']) != "p" and board(pos, square['x'] + 2, square['y'] - 1) != "p":
            return 1
    if board(pos, square['x'], square['y'] - 1) == "p":
        return 0
    lever = (1 if board(pos, square['x'] - 1, square['y'] - 1) == "p" else 0) + (1 if board(pos, square['x'] + 1, square['y'] - 1) == "p" else 0)
    leverpush = (1 if board(pos, square['x'] - 1, square['y'] - 2) == "p" else 0) + (1 if board(pos, square['x'] + 1, square['y'] - 2) == "p" else 0)
    phalanx = (1 if board(pos, square['x'] - 1, square['y']) == "P" else 0) + (1 if board(pos, square['x'] + 1, square['y']) == "P" else 0)
    if lever - supported(pos, square) > 1:
        return 0
    if leverpush - phalanx > 0:
        return 0
    if lever > 0 and leverpush > 0:
        return 0
    return 1

def king_proximity(pos, square = None):
    if square is None:
        return sum(pos, king_proximity)
    if not passed_leverable(pos, square):
        return 0
    r = rank(pos, square) - 1
    w = 5 * r - 13 if r > 2 else 0
    v = 0
    if w <= 0:
        return 0
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "k":
                v += int(min(max(abs(y - square['y'] + 1), abs(x - square['x'])), 5) * 19 / 4) << 0 * w
            if board(pos, x, y) == "K":
                v -= min(max(abs(y - square['y'] + 1), abs(x - square['x'])), 5) * 2 * w
                if square['y'] > 1:
                    v -= min(max(abs(y - square['y'] + 2), abs(x - square['x'])), 5) * w
    return v

def passed_block(pos, square = None):
    if square is None:
        return sum(pos, passed_block)
    if not passed_leverable(pos, square):
        return 0
    if rank(pos, square) < 4:
        return 0
    if board(pos, square['x'], square['y'] - 1) != "-":
        return 0
    r = rank(pos, square) - 1
    w = 5 * r - 13 if r > 2 else 0
    pos2 = colorflip(pos)
    defended = 0
    unsafe = 0
    wunsafe = 0
    defended1 = 0
    unsafe1 = 0
    for y in range(square['y'] - 1, -1, -1):
        if attack(pos, {'x': square['x'], 'y': y}):
            defended += 1
        if attack(pos2, {'x': square['x'], 'y': 7 - y}):
            unsafe += 1
        if attack(pos2, {'x': square['x'] - 1, 'y': 7 - y}):
            wunsafe += 1
        if attack(pos2, {'x': square['x'] + 1, 'y': 7 - y}):
            wunsafe += 1
        if y == square['y'] - 1:
            defended1 = defended
            unsafe1 = unsafe
    for y in range(square['y'] + 1, 8):
        if board(pos, square['x'], y) == "R" or board(pos, square['x'], y) == "Q":
            defended1 = defended = square['y']
        if board(pos, square['x'], y) == "r" or board(pos, square['x'], y) == "q":
            unsafe1 = unsafe = square['y']
    k = (35 if unsafe == 0 and wunsafe == 0 else 20 if unsafe == 0 else 9 if unsafe1 == 0 else 0) + (5 if defended1 != 0 else 0)
    return k * w

def passed_file(pos, square = None):
    if square is None:
        return sum(pos, passed_file)
    if not passed_leverable(pos, square):
        return 0
    file_val = ffile(pos, square)
    return min(file_val - 1, 8 - file_val)

def passed_rank(pos, square = None):
    if square is None:
        return sum(pos, passed_rank)
    if not passed_leverable(pos, square):
        return 0
    return rank(pos, square) - 1

def passed_leverable(pos, square = None):
    if square is None:
        return sum(pos, passed_leverable)
    if not candidate_passed(pos, square):
        return 0
    if board(pos, square['x'], square['y'] - 1) != "p":
        return 1
    pos2 = colorflip(pos)
    for i in range(-1, 2, 2):
        s1 = {"x": square['x'] + i, "y": square['y']}
        s2 = {"x": square['x'] + i, "y": 7 - square['y']}
        if (
            board(pos, square['x'] + i, square['y'] + 1) == "P"
            and board(pos, square['x'] + i, square['y']) not in "pnbrqk"
            and (attack(pos, s1) > 0 or attack(pos2, s2) <= 1)
        ):
            return 1
    return 0

def passed_mg(pos, square = None):
    if square is None:
        return sum(pos, passed_mg)
    if not passed_leverable(pos, square):
        return 0
    v = 0
    v += [0, 10, 17, 15, 62, 168, 276][passed_rank(pos, square)]
    v += passed_block(pos, square)
    v -= 11 * passed_file(pos, square)
    return v

def passed_eg(pos, square = None):
    if square is None:
        return sum(pos, passed_eg)
    if not passed_leverable(pos, square):
        return 0
    v = 0
    v += king_proximity(pos, square)
    v += [0, 28, 33, 41, 72, 177, 260][passed_rank(pos, square)]
    v += passed_block(pos, square)
    v -= 8 * passed_file(pos, square)
    return v


# END PASSED PAWNS

# BEGIN PAWNS

def isolated(pos, square = None):
    if square is None:
        return sum(pos, isolated)
    if board(pos, square['x'], square['y']) != "P":
        return 0
    for y in range(8):
        if board(pos, square['x'] - 1, y) == "P":
            return 0
        if board(pos, square['x'] + 1, y) == "P":
            return 0
    return 1

def opposed(pos, square = None):
    if square is None:
        return sum(pos, opposed)
    if board(pos, square['x'], square['y']) != "P":
        return 0
    for y in range(square['y']):
        if board(pos, square['x'], y) == "p":
            return 1
    return 0

def phalanx(pos, square = None):
    if square is None:
        return sum(pos, phalanx)
    if board(pos, square['x'], square['y']) != "P":
        return 0
    if board(pos, square['x'] - 1, square['y']) == "P":
        return 1
    if board(pos, square['x'] + 1, square['y']) == "P":
        return 1
    return 0

def supported(pos, square = None):
    if square is None:
        return sum(pos, supported)
    if board(pos, square['x'], square['y']) != "P":
        return 0
    return (1 if board(pos, square['x'] - 1, square['y'] + 1) == "P" else 0) + (1 if board(pos, square['x'] + 1, square['y'] + 1) == "P" else 0)

def backward(pos, square = None):
    if square is None:
        return sum(pos, backward)
    if board(pos, square['x'], square['y']) != "P":
        return 0
    for y in range(square['y'], 8):
        if board(pos, square['x'] - 1, y) == "P" or board(pos, square['x'] + 1, y) == "P":
            return 0
    if board(pos, square['x'] - 1, square['y'] - 2) == "p" or board(pos, square['x'] + 1, square['y'] - 2) == "p" or board(pos, square['x'], square['y'] - 1) == "p":
        return 1
    return 0

def doubled(pos, square = None):
    if square is None:
        return sum(pos, doubled)
    if board(pos, square['x'], square['y']) != "P":
        return 0
    if board(pos, square['x'], square['y'] + 1) != "P":
        return 0
    if board(pos, square['x'] - 1, square['y'] + 1) == "P":
        return 0
    if board(pos, square['x'] + 1, square['y'] + 1) == "P":
        return 0
    return 1

def connected(pos, square = None):
    if square is None:
        return sum(pos, connected)
    if supported(pos, square) or phalanx(pos, square):
        return 1
    return 0

def connected_bonus(pos, square = None):
    if square is None:
        return sum(pos, connected_bonus)
    if not connected(pos, square):
        return 0
    seed = [0, 7, 8, 12, 29, 48, 86]
    op = opposed(pos, square)
    ph = phalanx(pos, square)
    su = supported(pos, square)
    bl = 1 if board(pos, square['x'], square['y'] - 1) == "p" else 0
    r = rank(pos, square)
    if r < 2 or r > 7:
        return 0
    return seed[r - 1] * (2 + ph - op) + 21 * su

def weak_unopposed_pawn(pos, square = None):
    if square is None:
        return sum(pos, weak_unopposed_pawn)
    if opposed(pos, square):
        return 0
    v = 0
    if isolated(pos, square):
        v += 1
    elif backward(pos, square):
        v += 1
    return v

def weak_lever(pos, square = None):
    if square is None:
        return sum(pos, weak_lever)
    if board(pos, square['x'], square['y']) != "P":
        return 0
    if board(pos, square['x'] - 1, square['y'] - 1) != "p":
        return 0
    if board(pos, square['x'] + 1, square['y'] - 1) != "p":
        return 0
    if board(pos, square['x'] - 1, square['y'] + 1) == "P":
        return 0
    if board(pos, square['x'] + 1, square['y'] + 1) == "P":
        return 0
    return 1

def blocked(pos, square = None):
    if square is None:
        return sum(pos, blocked)
    if board(pos, square['x'], square['y']) != "P":
        return 0
    if square['y'] != 2 and square['y'] != 3:
        return 0
    if board(pos, square['x'], square['y'] - 1) != "p":
        return 0
    return 4 - square['y']

def doubled_isolated(pos, square = None):
    if square is None:
        return sum(pos, doubled_isolated)
    if board(pos, square['x'], square['y']) != "P":
        return 0
    if isolated(pos, square):
        obe = 0
        eop = 0
        ene = 0
        for y in range(8):
            if y > square['y'] and board(pos, square['x'], y) == "P":
                obe += 1
            if y < square['y'] and board(pos, square['x'], y) == "p":
                eop += 1
            if board(pos, square['x'] - 1, y) == "p" or board(pos, square['x'] + 1, y) == "p":
                ene += 1
        if obe > 0 and ene == 0 and eop > 0:
            return 1
    return 0

def pawns_mg(pos, square = None):
    if square is None:
        return sum(pos, pawns_mg)
    v = 0
    if doubled_isolated(pos, square):
        v -= 11
    elif isolated(pos, square):
        v -= 5
    elif backward(pos, square):
        v -= 9
    v -= doubled(pos, square) * 11
    v += connected(pos, square) * connected_bonus(pos, square)
    v -= 13 * weak_unopposed_pawn(pos, square)
    v += [0, -11, -3][blocked(pos, square)]
    return v

def pawns_eg(pos, square = None):
    if square is None:
        return sum(pos, pawns_eg)
    v = 0
    if doubled_isolated(pos, square):
        v -= 56
    elif isolated(pos, square):
        v -= 15
    elif backward(pos, square):
        v -= 24
    v -= doubled(pos, square) * 56
    v += connected(pos, square) * connected_bonus(pos, square) * (rank(pos, square) - 3) // 4
    v -= 27 * weak_unopposed_pawn(pos, square)
    v -= 56 * weak_lever(pos, square)
    v += [0, -4, 4][blocked(pos, square)]
    return v

# END PAWNS

# BEGIN PIECES

def outpost(pos, square = None):
    if square is None:
        return sum(pos, outpost)
    if board(pos, square['x'], square['y']) != "N" and board(pos, square['x'], square['y']) != "B":
        return 0
    if not outpost_square(pos, square):
        return 0
    return 1

def outpost_square(pos, square = None):
    if square is None:
        return sum(pos, outpost_square)
    if rank(pos, square) < 4 or rank(pos, square) > 6:
        return 0
    if board(pos, square['x'] - 1, square['y'] + 1) != "P" and board(pos, square['x'] + 1, square['y'] + 1) != "P":
        return 0
    if pawn_attacks_span(pos, square):
        return 0
    return 1

def reachable_outpost(pos, square = None):
    if square is None:
        return sum(pos, reachable_outpost)
    if board(pos, square['x'], square['y']) != "B" and board(pos, square['x'], square['y']) != "N":
        return 0
    v = 0
    for x in range(8):
        for y in range(2, 5):
            if (board(pos, square['x'], square['y']) == "N" and (board(pos, x, y) == '-' or board(pos, x, y).islower() or "PNBRQK".index(board(pos, x, y)) < 0)
                and knight_attack(pos, {'x': x, 'y': y}, square)
                and outpost_square(pos, {'x': x, 'y': y})) or \
                (board(pos, square['x'], square['y']) == "B" and (board(pos, x, y) == '-' or board(pos, x, y).islower() or "PNBRQK".index(board(pos, x, y)) < 0)
                and bishop_xray_attack(pos, {'x': x, 'y': y}, square)
                and outpost_square(pos, {'x': x, 'y': y})):
                support = 2 if board(pos, x - 1, y + 1) == "P" or board(pos, x + 1, y + 1) == "P" else 1
                v = max(v, support)
    return v

def minor_behind_pawn(pos, square = None):
    if square is None:
        return sum(pos, minor_behind_pawn)
    if board(pos, square['x'], square['y']) != "B" and board(pos, square['x'], square['y']) != "N":
        return 0
    if board(pos, square['x'], square['y'] - 1).upper() != "P":
        return 0
    return 1

def bishop_pawns(pos, square = None):
    if square is None:
        return sum(pos, bishop_pawns)
    if board(pos, square['x'], square['y']) != "B":
        return 0
    c = (square['x'] + square['y']) % 2
    v = 0
    blocked = 0
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "P" and c == (x + y) % 2:
                v += 1
            if board(pos, x, y) == "P" and x > 1 and x < 6 and board(pos, x, y - 1) != "-":
                blocked += 1
    return v * (blocked + (0 if pawn_attack(pos, square) > 0 else 1))

def rook_on_file(pos, square = None):
    if square is None:
        return sum(pos, rook_on_file)
    if board(pos, square['x'], square['y']) != "R":
        return 0
    open = 1
    for y in range(8):
        if board(pos, square['x'], y) == "P":
            return 0
        if board(pos, square['x'], y) == "p":
            open = 0
    return open + 1

def trapped_rook(pos, square = None):
    if square is None:
        return sum(pos, trapped_rook)
    if board(pos, square['x'], square['y']) != "R":
        return 0
    if rook_on_file(pos, square):
        return 0
    if mobility(pos, square) > 3:
        return 0
    kx = 0
    ky = 0
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "K":
                kx = x
                ky = y
    if (kx < 4) != (square['x'] < kx):
        return 0
    return 1

def weak_queen(pos, square = None):
    if square is None:
        return sum(pos, weak_queen)
    if board(pos, square['x'], square['y']) != "Q":
        return 0
    for i in range(8):
        ix = (i + (i > 3)) % 3 - 1
        iy = (int((i + (i > 3)) / 3) << 0) - 1
        count = 0
        for d in range(1, 8):
            b = board(pos, square['x'] + d * ix, square['y'] + d * iy)
            if b == "r" and (ix == 0 or iy == 0) and count == 1:
                return 1
            if b == "b" and (ix != 0 and iy != 0) and count == 1:
                return 1
            if b != "-":
                count += 1
    return 0

def king_protector(pos, square = None):
    if square is None:
        return sum(pos, king_protector)
    if board(pos, square['x'], square['y']) != "N" and board(pos, square['x'], square['y']) != "B":
        return 0
    return king_distance(pos, square)

def long_diagonal_bishop(pos, square = None):
    if square is None:
        return sum(pos, long_diagonal_bishop)
    if board(pos, square['x'], square['y']) != "B":
        return 0
    if square['x'] - square['y'] != 0 and square['x'] - (7 - square['y']) != 0:
        return 0
    x1 = square['x']
    y1 = square['y']
    if min(x1, 7 - x1) > 2:
        return 0
    for i in range(min(x1, 7 - x1), 4):
        if board(pos, x1, y1) == "p":
            return 0
        if board(pos, x1, y1) == "P":
            return 0
        if x1 < 4:
            x1 += 1
        else:
            x1 -= 1
        if y1 < 4:
            y1 += 1
        else:
            y1 -= 1
    return 1

def outpost_total(pos, square = None):
    if square is None:
        return sum(pos, outpost_total)
    if board(pos, square['x'], square['y']) != "N" and board(pos, square['x'], square['y']) != "B":
        return 0
    knight = board(pos, square['x'], square['y']) == "N"
    reachable = 0
    if not outpost(pos, square):
        if not knight:
            return 0
        reachable = reachable_outpost(pos, square)
        if not reachable:
            return 0
        return 1
    if knight and (square['x'] < 2 or square['x'] > 5):
        ea = 0
        cnt = 0
        for x in range(8):
            for y in range(8):
                if board(pos, x, y) not in "nbrqk":
                    pass
                elif (abs(square['x'] - x) == 2 and abs(square['y'] - y) == 1 or abs(square['x'] - x) == 1 and abs(square['y'] - y) == 2):
                    ea = 1
                if board(pos, x, y) not in "nbrqk":
                    continue
                elif (x < 4 and square['x'] < 4 or x >= 4 and square['x'] >= 4):
                    cnt += 1
        if not ea and cnt <= 1:
            return 2
    return 4 if knight else 3

def rook_on_queen_file(pos, square = None):
    if square is None:
        return sum(pos, rook_on_queen_file)
    if board(pos, square['x'], square['y']) != "R":
        return 0
    for y in range(8):
        if board(pos, square['x'], y).upper() == "Q":
            return 1
    return 0

def bishop_xray_pawns(pos, square = None):
    if square is None:
        return sum(pos, bishop_xray_pawns)
    if board(pos, square['x'], square['y']) != "B":
        return 0
    count = 0
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "p" and abs(square['x'] - x) == abs(square['y'] - y):
                count += 1
    return count

def rook_on_king_ring(pos, square = None):
    if square is None:
        return sum(pos, rook_on_king_ring)
    if board(pos, square['x'], square['y']) != "R":
        return 0
    if king_attackers_count(pos, square) > 0:
        return 0
    for y in range(8):
        if king_ring(pos, {"x": square['x'], "y": y}, full = None):
            return 1
    return 0

def bishop_on_king_ring(pos, square = None):
    if square is None:
        return sum(pos, bishop_on_king_ring)
    if board(pos, square['x'], square['y']) != "B":
        return 0
    if king_attackers_count(pos, square) > 0:
        return 0
    for i in range(4):
        ix = ((i > 1) * 2 - 1)
        iy = ((i % 2 == 0) * 2 - 1)
        for d in range(1, 8):
            x = square['x'] + d * ix
            y = square['y'] + d * iy
            if board(pos, x, y) == "x":
                break
            if king_ring(pos, {"x": x, "y": y}, full = None):
                return 1
            if board(pos, x, y).upper() == "P":
                break
    return 0

def queen_infiltration(pos, square = None):
    if square is None:
        return sum(pos, queen_infiltration)
    if board(pos, square['x'], square['y']) != "Q":
        return 0
    if square['y'] > 3:
        return 0
    if board(pos, square['x'] + 1, square['y'] - 1) == "p":
        return 0
    if board(pos, square['x'] - 1, square['y'] - 1) == "p":
        return 0
    if pawn_attacks_span(pos, square):
        return 0
    return 1

def pieces_mg(pos, square = None):
    if square is None:
        return sum(pos, pieces_mg)
    if board(pos, square['x'], square['y']) not in "NBRQ":
        return 0
    v = 0
    v += [0,31,-7,30,56][outpost_total(pos, square)]
    v += 18 * minor_behind_pawn(pos, square)
    v -= 3 * bishop_pawns(pos, square)
    v -= 4 * bishop_xray_pawns(pos, square)
    v += 6 * rook_on_queen_file(pos, square)
    v += 16 * rook_on_king_ring(pos, square)
    v += 24 * bishop_on_king_ring(pos, square)
    v += [0,19,48][rook_on_file(pos, square)]
    v -= trapped_rook(pos, square) * 55 * (1 if pos['c'][0] or pos['c'][1] else 2)
    v -= 56 * weak_queen(pos, square)
    v -= 2 * queen_infiltration(pos, square)
    v -= (8 if board(pos, square['x'], square['y']) == "N" else 6) * king_protector(pos, square)
    v += 45 * long_diagonal_bishop(pos, square)
    return v

def pieces_eg(pos, square = None):
    if square is None:
        return sum(pos, pieces_eg)
    if board(pos, square['x'], square['y']) not in "NBRQ":
        return 0
    v = 0
    v += [0,22,36,23,36][outpost_total(pos, square)]
    v += 3 * minor_behind_pawn(pos, square)
    v -= 7 * bishop_pawns(pos, square)
    v -= 5 * bishop_xray_pawns(pos, square)
    v += 11 * rook_on_queen_file(pos, square)
    v += [0,7,29][rook_on_file(pos, square)]
    v -= trapped_rook(pos, square) * 13 * (1 if pos['c'][0] or pos['c'][1] else 2)
    v -= 15 * weak_queen(pos, square)
    v += 14 * queen_infiltration(pos, square)
    v -= 9 * king_protector(pos, square)
    return v

# END PIECES

# BEGIN SPACE

def space_area(pos, square = None):
    if square is None:
        return sum(pos, space_area)
    v = 0
    rank = rank(pos, square)
    ffile = ffile(pos, square)
    if (rank >= 2 and rank <= 4 and ffile >= 3 and ffile <= 6) and (board(pos, square['x'] ,square['y']) != "P") and (board(pos, square['x'] - 1 ,square['y'] - 1) != "p") and (board(pos, square['x'] + 1 ,square['y'] - 1) != "p"):
        v += 1
        if (board(pos, square['x'], square['y'] - 1) == "P" or board(pos, square['x'], square['y'] - 2) == "P" or board(pos, square['x'], square['y'] - 3) == "P") and not attack(colorflip(pos), {"x":square['x'], "y":7-square['y']}):
            v += 1
    return v

def space(pos, square = None):
    if non_pawn_material(pos) + non_pawn_material(colorflip(pos)) < 12222:
        return 0
    pieceCount = 0
    blockedCount = 0
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) in "PNBRQK":
                pieceCount += 1
            if board(pos, x, y) == "P" and (board(pos, x, y - 1) == "p" or (board(pos, x - 1, y - 2) == "p" and board(pos, x + 1, y - 2) == "p")):
                blockedCount += 1
            if board(pos, x, y) == "p" and (board(pos, x, y + 1) == "P" or (board(pos, x - 1, y + 2) == "P" and board(pos, x + 1, y + 2) == "P")):
                blockedCount += 1
    weight = pieceCount - 3 + min(blockedCount, 9)
    return int((space_area(pos, square) * weight * weight / 16))

# END SPACE

# BEGIN THREATS

def safe_pawn(pos, square = None):
    if square is None:
        return sum(pos, safe_pawn)
    if board(pos, square['x'], square['y']) != "P":
        return 0
    if attack(pos, square):
        return 1
    if not attack(colorflip(pos), {"x": square['x'], "y": 7 - square['y']}):
        return 1
    return 0

def threat_safe_pawn(pos, square = None):
    if square is None:
        return sum(pos, threat_safe_pawn)
    if board(pos, square['x'], square['y']) not in "nbrq":
        return 0
    if not pawn_attack(pos, square):
        return 0
    if safe_pawn(pos, {"x": square['x'] - 1, "y": square['y'] + 1}) or safe_pawn(pos, {"x": square['x'] + 1, "y": square['y'] + 1}):
        return 1
    return 0

def weak_enemies(pos, square = None):
    if square is None:
        return sum(pos, weak_enemies)
    if board(pos, square['x'], square['y']) not in "pnbrqk":
        return 0
    if board(pos, square['x'] - 1, square['y'] - 1) == "p":
        return 0
    if board(pos, square['x'] + 1, square['y'] - 1) == "p":
        return 0
    if not attack(pos, square):
        return 0
    if attack(pos, square) <= 1 and attack(colorflip(pos), {"x": square['x'], "y": 7 - square['y']}) > 1:
        return 0
    return 1

def minor_threat(pos, square = None):
    if square is None:
        return sum(pos, minor_threat)

    if board(pos, square['x'], square['y']) not in ['p', 'n', 'b', 'r', 'q', 'k']:
        return 0

    type_piece = "pnbrqk".index(board(pos, square['x'], square['y']))
    if type_piece < 0:
        return 0
    if not knight_attack(pos, s2 = None, square = square) and not bishop_xray_attack(pos, s2 = None, square = square):
        return 0
    if (board(pos, square['x'], square['y']) == "p"
        or not (board(pos, square['x'] - 1, square['y'] - 1) == "p"
                or board(pos, square['x'] + 1, square['y'] - 1) == "p"
                or (attack(pos, square) <= 1 and attack(colorflip(pos), {"x": square['x'], "y": 7 - square['y']}) > 1))
        and not weak_enemies(pos, square)):
        return 0
    return type_piece + 1

def rook_threat(pos, square = None):
    if square is None:
        return sum(pos, rook_threat)
    if board(pos, square['x'], square['y']) not in ['p', 'n', 'b', 'r', 'q', 'k']:
        return 0
    type_piece = "pnbrqk".index(board(pos, square['x'], square['y']))
    if type_piece < 0:
        return 0
    if not weak_enemies(pos, square):
        return 0
    if not rook_xray_attack(pos, s2 = None, square = square):
        return 0
    return type_piece + 1

def hanging(pos, square = None):
    if square is None:
        return sum(pos, hanging)
    if not weak_enemies(pos, square):
        return 0
    if board(pos, square['x'], square['y']) != "p" and attack(pos, square) > 1:
        return 1
    if not attack(colorflip(pos), {'x': square['x'], 'y': 7 - square['y']}):
        return 1
    return 0

def king_threat(pos, square = None):
    if square is None:
        return sum(pos, king_threat)
    if "pnbrq".find(board(pos, square['x'], square['y'])) < 0:
        return 0
    if not weak_enemies(pos, square):
        return 0
    if not king_attack(pos, square):
        return 0
    return 1

def pawn_push_threat(pos, square = None):
    if square is None:
        return sum(pos, pawn_push_threat)
    if "pnbrqk".find(board(pos, square['x'], square['y'])) < 0:
        return 0
    for ix in range(-1, 2, 2):
        if (board(pos, square['x'] + ix, square['y'] + 2) == "P" and
            board(pos, square['x'] + ix, square['y'] + 1) == "-" and
            board(pos, square['x'] + ix - 1, square['y']) != "p" and
            board(pos, square['x'] + ix + 1, square['y']) != "p" and
            (attack(pos, {'x': square['x'] + ix, 'y': square['y'] + 1}) or
             not attack(colorflip(pos), {'x': square['x'] + ix, 'y': 6 - square['y']}))):
            return 1
        if (square['y'] == 3 and
            board(pos, square['x'] + ix, square['y'] + 3) == "P" and
            board(pos, square['x'] + ix, square['y'] + 2) == "-" and
            board(pos, square['x'] + ix, square['y'] + 1) == "-" and
            board(pos, square['x'] + ix - 1, square['y']) != "p" and
            board(pos, square['x'] + ix + 1, square['y']) != "p" and
            (attack(pos, {'x': square['x'] + ix, 'y': square['y'] + 1}) or
             not attack(colorflip(pos), {'x': square['x'] + ix, 'y': 6 - square['y']}))):
            return 1
    return 0

def slider_on_queen(pos, square = None):
    if square is None:
        return sum(pos, slider_on_queen)
    pos2 = colorflip(pos)
    if queen_count(pos2) != 1:
        return 0
    if board(pos, square['x'], square['y']) == "P":
        return 0
    if board(pos, square['x'] - 1, square['y'] - 1) == "p":
        return 0
    if board(pos, square['x'] + 1, square['y'] - 1) == "p":
        return 0
    if attack(pos, square) <= 1:
        return 0
    if not mobility_area(pos, square):
        return 0
    diagonal = queen_attack_diagonal(pos2, square = {'x': square['x'], 'y': 7 - square['y']})
    v = 2 if queen_count(pos) == 0 else 1
    if diagonal and bishop_xray_attack(pos, s2 = None, square = square):
        return v
    if (not diagonal
        and rook_xray_attack(pos, s2 = None, square = square)
        and queen_attack(pos2, {'x': square['x'], 'y': 7 - square['y']})):
        return v
    return 0

def knight_on_queen(pos, square=None):
    if square is None:
        return sum(pos, knight_on_queen)
    pos2 = colorflip(pos)
    qx, qy = -1, -1
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "q":
                if qx >= 0 or qy >= 0:
                    return 0
                qx = x
                qy = y
    if queen_count(pos2) != 1:
        return 0
    if board(pos, square['x'], square['y']) == "P":
        return 0
    if board(pos, square['x'] - 1, square['y'] - 1) == "p":
        return 0
    if board(pos, square['x'] + 1, square['y'] - 1) == "p":
        return 0
    if attack(pos, square) <= 1 and attack(pos2, {'x': square['x'], 'y': 7 - square['y']}) > 1:
        return 0
    if not mobility_area(pos, square):
        return 0
    if not knight_attack(pos, s2 = None, square = square):
        return 0
    v = 2 if queen_count(pos) == 0 else 1
    if abs(qx - square['x']) == 2 and abs(qy - square['y']) == 1:
        return v
    if abs(qx - square['x']) == 1 and abs(qy - square['y']) == 2:
        return v
    return 0

def restricted(pos, square=None):
    if square is None:
        return sum(pos, restricted)
    if attack(pos, square) == 0:
        return 0
    pos2 = colorflip(pos)
    if not attack(pos2, {'x': square['x'], 'y': 7 - square['y']}):
        return 0
    if pawn_attack(pos2, {'x': square['x'], 'y': 7 - square['y']}) > 0:
        return 0
    if attack(pos2, {'x': square['x'], 'y': 7 - square['y']}) > 1 and attack(pos, square) == 1:
        return 0
    return 1

def weak_queen_protection(pos, square=None):
    if square is None:
        return sum(pos, weak_queen_protection)
    if not weak_enemies(pos, square):
        return 0
    if not queen_attack(colorflip(pos), {'x': square['x'], 'y': 7 - square['y']}):
        return 0
    return 1

def threats_mg(pos):
    v = 0
    v += 69 * hanging(pos)
    v += 24 if king_threat(pos) > 0 else 0
    v += 48 * pawn_push_threat(pos)
    v += 173 * threat_safe_pawn(pos)
    v += 60 * slider_on_queen(pos)
    v += 16 * knight_on_queen(pos)
    v += 7 * restricted(pos)
    v += 14 * weak_queen_protection(pos)
    for x in range(8):
        for y in range(8):
            s = {'x': x, 'y': y}
            v += [0, 5, 57, 77, 88, 79, 0][minor_threat(pos, s)]
            v += [0, 3, 37, 42, 0, 58, 0][rook_threat(pos, s)]
    return v

def threats_eg(pos):
    v = 0
    v += 36 * hanging(pos)
    v += 89 if king_threat(pos) > 0 else 0
    v += 39 * pawn_push_threat(pos)
    v += 94 * threat_safe_pawn(pos)
    v += 18 * slider_on_queen(pos)
    v += 11 * knight_on_queen(pos)
    v += 7 * restricted(pos)
    for x in range(8):
        for y in range(8):
            s = {'x': x, 'y': y}
            v += [0, 32, 41, 56, 119, 161, 0][minor_threat(pos, s)]
            v += [0, 46, 68, 60, 38, 41, 0][rook_threat(pos, s)]
    return v

######### END THREATS


######### BEGIN WINNABLE

def winnable(pos, square = None):
    if square is not None:
        return 0
    pawns = 0
    kx = [0, 0]
    ky = [0, 0]
    flanks = [0, 0]
    for x in range(8):
        open = [0, 0]
        for y in range(8):
            if board(pos, x, y).upper() == "P":
                open[0 if board(pos, x, y) == "P" else 1] = 1
                pawns += 1
            if board(pos, x, y).upper() == "K":
                kx[0 if board(pos, x, y) == "K" else 1] = x
                ky[0 if board(pos, x, y) == "K" else 1] = y
        if open[0] + open[1] > 0:
            flanks[0 if x < 4 else 1] = 1
    pos2 = colorflip(pos)
    passedCount = candidate_passed(pos) + candidate_passed(pos2)
    bothFlanks = 1 if flanks[0] and flanks[1] else 0
    outflanking = abs(kx[0] - kx[1]) - abs(ky[0] - ky[1])
    purePawn = 1 if (non_pawn_material(pos) + non_pawn_material(pos2)) == 0 else 0
    almostUnwinnable = outflanking < 0 and bothFlanks == 0
    infiltration = 1 if ky[0] < 4 or ky[1] > 3 else 0
    return (9 * passedCount
            + 12 * pawns
            + 9 * outflanking
            + 21 * bothFlanks
            + 24 * infiltration
            + 51 * purePawn
            - 43 * almostUnwinnable
            - 110)


def winnable_total_mg(pos, v=None):
    if v is None:
        v = middle_game_evaluation(pos, True)
    return (1 if v > 0 else -1 if v < 0 else 0) * max(min(winnable(pos) + 50, 0), -abs(v))


def winnable_total_eg(pos, v=None):
    if v is None:
        v = end_game_evaluation(pos, True)
    return (1 if v > 0 else -1 if v < 0 else 0) * max(winnable(pos), -abs(v))

############# END WINNABLE




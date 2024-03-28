# MAIN EVALUATIONS

def main_evaluation(pos):
    mg = middle_game_evaluation(pos)
    eg = end_game_evaluation(pos)
    p = phase(pos)
    rule50 = rule50(pos)
    eg = eg * scale_factor(pos, eg) / 64
    v = (((mg * p + ((eg * (128 - p)) // 1)) // 128) // 1)
    if len(arguments) == 1:
        v = ((v // 16) // 1) * 16
    v += tempo(pos)
    v = (v * (100 - rule50) // 100) // 1
    return v

def middle_game_evaluation(pos, nowinnable):
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

def end_game_evaluation(pos, nowinnable):
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

def pinned_direction(pos, square):
    if square is None:
        return sum(pos, pinned_direction)
    if board(pos, square.x, square.y).upper() not in "PNBRQK":
        return 0
    color = 1
    if board(pos, square.x, square.y) not in "PNBRQK":
        color = -1
    for i in range(8):
        ix = (i + (i > 3)) % 3 - 1
        iy = (((i + (i > 3)) // 3) << 0) - 1
        king = False
        for d in range(1, 8):
            b = board(pos, square.x + d * ix, square.y + d * iy)
            if b == "K":
                king = True
            if b != "-":
                break
        if king:
            for d in range(1, 8):
                b = board(pos, square.x - d * ix, square.y - d * iy)
                if b == "q" or (b == "b" and ix * iy != 0) or (b == "r" and ix * iy == 0):
                    return abs(ix + iy * 3) * color
                if b != "-":
                    break
    return 0

def knight_attack(pos, square, s2):
    if square is None:
        return sum(pos, knight_attack)
    v = 0
    for i in range(8):
        ix = ((i > 3) + 1) * (((i % 4) > 1) * 2 - 1)
        iy = (2 - (i > 3)) * ((i % 2 == 0) * 2 - 1)
        b = board(pos, square.x + ix, square.y + iy)
        if b == "N" and (s2 is None or s2.x == square.x + ix and s2.y == square.y + iy) and not pinned(pos, {"x": square.x + ix, "y": square.y + iy}):
            v += 1
    return v

def bishop_xray_attack(pos, square, s2):
    if square is None:
        return sum(pos, bishop_xray_attack)
    v = 0
    for i in range(4):
        ix = ((i > 1) * 2 - 1)
        iy = ((i % 2 == 0) * 2 - 1)
        for d in range(1, 8):
            b = board(pos, square.x + d * ix, square.y + d * iy)
            if b == "B" and (s2 is None or (s2.x == square.x + d * ix and s2.y == square.y + d * iy)):
                dir = pinned_direction(pos, {"x": square.x + d * ix, "y": square.y + d * iy})
                if dir == 0 or abs(ix + iy * 3) == dir:
                    v += 1
            if b != "-" and b != "Q" and b != "q":
                break
    return v

def rook_xray_attack(pos, square, s2):
    if square is None:
        return sum(pos, rook_xray_attack)
    v = 0
    for i in range(4):
        ix = -1 if i == 0 else 1 if i == 1 else 0
        iy = -1 if i == 2 else 1 if i == 3 else 0
        for d in range(1, 8):
            b = board(pos, square.x + d * ix, square.y + d * iy)
            if b == "R" and (s2 is None or (s2.x == square.x + d * ix and s2.y == square.y + d * iy)):
                dir = pinned_direction(pos, {"x": square.x + d * ix, "y": square.y + d * iy})
                if dir == 0 or abs(ix + iy * 3) == dir:
                    v += 1
            if b != "-" and b != "R" and b != "Q" and b != "q":
                break
    return v

def queen_attack(pos, square, s2):
    if square is None:
        return sum(pos, queen_attack)
    v = 0
    for i in range(8):
        ix = (i + (i > 3)) % 3 - 1
        iy = (((i + (i > 3)) // 3) << 0) - 1
        for d in range(1, 8):
            b = board(pos, square.x + d * ix, square.y + d * iy)
            if b == "Q" and (s2 is None or (s2.x == square.x + d * ix and s2.y == square.y + d * iy)):
                dir = pinned_direction(pos, {"x": square.x + d * ix, "y": square.y + d * iy})
                if dir == 0 or abs(ix + iy * 3) == dir:
                    v += 1
            if b != "-":
                break
    return v


def pawn_attack(pos, square):
    if square is None:
        return sum(pos, pawn_attack)
    v = 0
    if board(pos, square.x - 1, square.y + 1) == "P":
        v += 1
    if board(pos, square.x + 1, square.y + 1) == "P":
        v += 1
    return v

def king_attack(pos, square):
    if square is None:
        return sum(pos, king_attack)
    for i in range(8):
        ix = (i + (i > 3)) % 3 - 1
        iy = (((i + (i > 3)) // 3) << 0) - 1
        if board(pos, square.x + ix, square.y + iy) == "K":
            return 1
    return 0

def attack(pos, square):
    if square is None:
        return sum(pos, attack)
    v = 0
    v += pawn_attack(pos, square)
    v += king_attack(pos, square)
    v += knight_attack(pos, square)
    v += bishop_xray_attack(pos, square)
    v += rook_xray_attack(pos, square)
    v += queen_attack(pos, square)
    return v

def queen_attack_diagonal(pos, square, s2):
    if square is None:
        return sum(pos, queen_attack_diagonal)
    v = 0
    for i in range(8):
        ix = (i + (i > 3)) % 3 - 1
        iy = (((i + (i > 3)) // 3) << 0) - 1
        if ix == 0 or iy == 0:
            continue
        for d in range(1, 8):
            b = board(pos, square.x + d * ix, square.y + d * iy)
            if b == "Q" and (s2 is None or s2.x == square.x + d * ix and s2.y == square.y + d * iy):
                dir = pinned_direction(pos, {"x": square.x + d * ix, "y": square.y + d * iy})
                if dir == 0 or abs(ix + iy * 3) == dir:
                    v += 1
            if b != "-":
                break
    return v

def pinned(pos, square):
    if square is None:
        return sum(pos, pinned)
    if board(pos, square.x, square.y) not in "PNBRQK":
        return 0
    return 1 if pinned_direction(pos, square) > 0 else 0

# END ATTACK

# BEGIN HELPERS

def rank(pos, square):
    if square is None:
        return sum(pos, rank)
    return 8 - square.y

def file(pos, square):
    if square is None:
        return sum(pos, file)
    return 1 + square.x

def bishop_count(pos, square):
    if square is None:
        return sum(pos, bishop_count)
    if board(pos, square.x, square.y) == "B":
        return 1
    return 0

def queen_count(pos, square):
    if square is None:
        return sum(pos, queen_count)
    if board(pos, square.x, square.y) == "Q":
        return 1
    return 0

def pawn_count(pos, square):
    if square is None:
        return sum(pos, pawn_count)
    if board(pos, square.x, square.y) == "P":
        return 1
    return 0

def knight_count(pos, square):
    if square is None:
        return sum(pos, knight_count)
    if board(pos, square.x, square.y) == "N":
        return 1
    return 0

def rook_count(pos, square):
    if square is None:
        return sum(pos, rook_count)
    if board(pos, square.x, square.y) == "R":
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

def king_distance(pos, square):
    if square is None:
        return sum(pos, king_distance)
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "K":
                return max(abs(x - square.x), abs(y - square.y))
    return 0

def king_ring(pos, square, full):
    if square is None:
        return sum(pos, king_ring)
    if not full and board(pos, square.x + 1, square.y - 1) == "p" and board(pos, square.x - 1, square.y - 1) == "p":
        return 0
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            if board(pos, square.x + ix, square.y + iy) == "k" and (ix >= -1 and ix <= 1 or square.x + ix == 0 or square.x + ix == 7) and (iy >= -1 and iy <= 1 or square.y + iy == 0 or square.y + iy == 7):
                return 1
    return 0

def piece_count(pos, square):
    if square is None:
        return sum(pos, piece_count)
    i = "PNBRQK".index(board(pos, square.x, square.y))
    return 1 if i >= 0 else 0

def pawn_attacks_span(pos, square):
    if square is None:
        return sum(pos, pawn_attacks_span)
    pos2 = colorflip(pos)
    for y in range(square.y):
        if board(pos, square.x - 1, y) == "p" and (y == square.y - 1 or (board(pos, square.x - 1, y + 1) != "P" and not backward(pos2, {x:square.x-1, y:7-y}))):
            return 1
        if board(pos, square.x + 1, y) == "p" and (y == square.y - 1 or (board(pos, square.x + 1, y + 1) != "P" and not backward(pos2, {x:square.x+1, y:7-y}))):
            return 1
    return 0


# END HELPERS

# BEGIN IMBALANCE

def imbalance(pos, square):
    if square is None:
        return sum(pos, imbalance)
    qo = [[0],[40,38],[32,255,-62],[0,104,4,0],[-26,-2,47,105,-208],[-189,24,117,133,-134,-6]]
    qt = [[0],[36,0],[9,63,0],[59,65,42,0],[46,39,24,-24,0],[97,100,-42,137,268,0]]
    j = "XPNBRQxpnbrq".index(board(pos, square.x, square.y))
    if j < 0 or j > 5:
        return 0
    bishop = [0, 0]
    v = 0
    for x in range(8):
        for y in range(8):
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

def bishop_pair(pos, square):
    if bishop_count(pos) < 2:
        return 0
    if square is None:
        return 1438
    return 1 if board(pos, square.x, square.y) == "B" else 0

def imbalance_total(pos, square):
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

def strength_square(pos, square):
    if square == None:
        return sum(pos, strength_square)
    v = 5
    kx = min(6, max(1, square.x))
    weakness = [[-6,81,93,58,39,18,25],
                [-43,61,35,-49,-29,-11,-63],
                [-10,75,23,-2,32,3,-45],
                [-39,-13,-29,-52,-48,-67,-166]]
    for x in range(kx - 1, kx + 2):
        us = 0
        for y in range(7, square.y - 1, -1):
            if board(pos, x, y) == "p" and board(pos, x-1, y+1) != "P" and board(pos, x+1, y+1) != "P":
                us = y
        f = min(x, 7 - x)
        v += weakness[f][us] if weakness[f][us] else 0
    return v

def storm_square(pos, square, eg):
    if square == None:
        return sum(pos, storm_square)
    v = 0
    ev = 5
    kx = min(6, max(1, square.x))
    unblockedstorm = [[85,-289,-166,97,50,45,50],
                      [46,-25,122,45,37,-10,20],
                      [-6,51,168,34,-2,-22,-14],
                      [-15,-11,101,4,11,-15,-29]]
    blockedstorm = [[0,0,76,-10,-7,-4,-1],
                    [0,0,78,15,10,6,2]]
    for x in range(kx - 1, kx + 2):
        us = 0
        them = 0
        for y in range(7, square.y - 1, -1):
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

def shelter_strength(pos, square):
    w = 0
    s = 1024
    tx = None
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "k" or (pos.c[2] and x == 6 and y == 0) or (pos.c[3] and x == 2 and y == 0):
                w1 = strength_square(pos, {"x":x,"y":y})
                s1 = storm_square(pos, {"x":x,"y":y})
                if s1 - w1 < s - w:
                    w = w1
                    s = s1
                    tx = max(1, min(6, x))
    if square == None:
        return w
    if tx != None and board(pos, square.x, square.y) == "p" and square.x >= tx-1 and square.x <= tx+1:
        for y in range(square.y-1, -1, -1):
            if board(pos, square.x, y) == "p":
                return 0
        return 1
    return 0

def shelter_storm(pos, square):
    w = 0
    s = 1024
    tx = None
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "k" or (pos.c[2] and x == 6 and y == 0) or (pos.c[3] and x == 2 and y == 0):
                w1 = strength_square(pos, {"x": x, "y": y})
                s1 = storm_square(pos, {"x": x, "y": y})
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

def king_pawn_distance(pos, square):
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

def check(pos, square, type):
    if square is None:
        return sum(pos, check)
    if (rook_xray_attack(pos, square) and (type is None or type == 2 or type == 4)) or (queen_attack(pos, square) and (type is None or type == 3)):
        for i in range(4):
            ix = -1 if i == 0 else 1 if i == 1 else 0
            iy = -1 if i == 2 else 1 if i == 3 else 0
            for d in range(1, 8):
                b = board(pos, square["x"] + d * ix, square["y"] + d * iy)
                if b == "k":
                    return 1
                if b != "-" and b != "q":
                    break
    if (bishop_xray_attack(pos, square) and (type is None or type == 1 or type == 4)) or (queen_attack(pos, square) and (type is None or type == 3)):
        for i in range(4):
            ix = (2 * (i > 1) - 1)
            iy = (2 * (i % 2 == 0) - 1)
            for d in range(1, 8):
                b = board(pos, square["x"] + d * ix, square["y"] + d * iy)
                if b == "k":
                    return 1
                if b != "-" and b != "q":
                    break
    if knight_attack(pos, square) and (type is None or type == 0 or type == 4):
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

def safe_check(pos, square, type):
    if square is None:
        return sum(pos, safe_check, type)
    if board(pos, square.x, square.y) is None:
        return 0
    if "PNBRQK".index(board(pos, square.x, square.y)) >= 0:
        return 0
    if not check(pos, square, type):
        return 0
    pos2 = colorflip(pos)
    if type == 3 and safe_check(pos, square, 2):
        return 0
    if type == 1 and safe_check(pos, square, 3):
        return 0
    if (not attack(pos2, {"x": square.x, "y": 7 - square.y})
        or (weak_squares(pos, square) and attack(pos, square) > 1))
        and (type != 3 or not queen_attack(pos2, {"x": square.x, "y": 7 - square.y})):
        return 1
    return 0

def king_attackers_count(pos, square):
    if square is None:
        return sum(pos, king_attackers_count)
    if board(pos, square.x, square.y) is None:
        return 0
    if board(pos, square.x, square.y) == "P":
        v = 0
        for dir in range(-1, 2, 2):
            fr = board(pos, square.x + dir * 2, square.y) == "P"
            if square.x + dir >= 0 and square.x + dir <= 7 and king_ring(pos, {"x": square.x + dir, "y": square.y - 1}, True):
                v = v + (fr ? 0.5 : 1)
        return v
    for x in range(8):
        for y in range(8):
            s2 = {"x": x, "y": y}
            if king_ring(pos, s2):
                if (knight_attack(pos, s2, square)
                    or bishop_xray_attack(pos, s2, square)
                    or rook_xray_attack(pos, s2, square)
                    or queen_attack(pos, s2, square)):
                    return 1
    return 0

def king_attackers_weight(pos, square):
    if square is None:
        return sum(pos, king_attackers_weight)
    if king_attackers_count(pos, square):
        return [0, 81, 52, 44, 10]["PNBRQ".index(board(pos, square.x, square.y))]
    return 0

def king_attacks(pos, square):
    if square is None:
        return sum(pos, king_attacks)
    if board(pos, square.x, square.y) is None:
        return 0
    if "NBRQ".index(board(pos, square.x, square.y)) < 0:
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
                v += knight_attack(pos, s2, square)
                v += bishop_xray_attack(pos, s2, square)
                v += rook_xray_attack(pos, s2, square)
                v += queen_attack(pos, s2, square)
    return v

def weak_bonus(pos, square):
    if square is None:
        return sum(pos, weak_bonus)
    if not weak_squares(pos, square):
        return 0
    if not king_ring(pos, square):
        return 0
    return 1

def weak_squares(pos, square):
    if square is None:
        return sum(pos, weak_squares)
    if attack(pos, square):
        pos2 = colorflip(pos)
        attack = attack(pos2, {'x': square['x'], 'y': 7 - square['y']})
        if attack >= 2:
            return 0
        if attack == 0:
            return 1
        if king_attack(pos2, {'x': square['x'], 'y': 7 - square['y']}) or queen_attack(pos2, {'x': square['x'], 'y': 7 - square['y']}):
            return 1
    return 0

def unsafe_checks(pos, square):
    if square is None:
        return sum(pos, unsafe_checks)
    if check(pos, square, 0) and safe_check(pos, None, 0) == 0:
        return 1
    if check(pos, square, 1) and safe_check(pos, None, 1) == 0:
        return 1
    if check(pos, square, 2) and safe_check(pos, None, 2) == 0:
        return 1
    return 0

def knight_defender(pos, square):
    if square is None:
        return sum(pos, knight_defender)
    if knight_attack(pos, square) and king_attack(pos, square):
        return 1
    return 0

def endgame_shelter(pos, square):
    w = 0
    s = 1024
    tx = None
    for x in range(8):
        for y in range(8):
            if board(pos, x, y) == "k" or (pos['c'][2] and x == 6 and y == 0) or (pos['c'][3] and x == 2 and y == 0):
                w1 = strength_square(pos, {'x': x, 'y': y})
                s1 = storm_square(pos, {'x': x, 'y': y})
                e1 = storm_square(pos, {'x': x, 'y': y}, True)
                if s1 - w1 < s - w:
                    w = w1
                    s = s1
                    e = e1
    if square is None:
        return e
    return 0

def blockers_for_king(pos, square):
    if square is None:
        return sum(pos, blockers_for_king)
    if pinned_direction(colorflip(pos), {'x': square['x'], 'y': 7 - square['y']}):
        return 1
    return 0

def flank_attack(pos, square):
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

def flank_defense(pos, square):
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
    noQueen = (queen_count(pos) > 0) ? 0 : 1
    v = count * weight + 69 * kingAttacks + 185 * weak - 100 * (knight_defender(colorflip(pos)) > 0) + 148 * unsafeChecks + 98 * blockersForKing - 4 * kingFlankDefense + ((3 * kingFlankAttack * kingFlankAttack / 8) << 0) - 873 * noQueen - ((6 * (shelter_strength(pos) - shelter_storm(pos)) / 8) << 0) + mobility_mg(pos) - mobility_mg(colorflip(pos)) + 37 + ((772 * min(safe_check(pos, None, 3), 1.45)) << 0) + ((1084 * min(safe_check(pos, None, 2), 1.75)) << 0) + ((645 * min(safe_check(pos, None, 1), 1.50)) << 0) + ((792 * min(safe_check(pos, None, 0), 1.62)) << 0)
    if v > 100:
        return v
    return 0

def king_mg(pos):
    v = 0
    kd = king_danger(pos)
    v -= shelter_strength(pos)
    v += shelter_storm(pos)
    v += (kd * kd / 4096) << 0
    v += 8 * flank_attack(pos)
    v += 17 * pawnless_flank(pos)
    return v

def king_eg(pos):
    v = 0
    v -= 16 * king_pawn_distance(pos)
    v += endgame_shelter(pos)
    v += 95 * pawnless_flank(pos)
    v += (king_danger(pos) / 16) << 0
    return v


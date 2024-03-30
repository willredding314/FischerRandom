# BOARD REPRESENTATION

def board(pos, x, y):
    if x >= 0 and x <= 7 and y >= 0 and y <= 7:
        return pos['b'][x][y]
    return "x"

def colorflip(pos):
    board = [[None] * 8 for _ in range(8)]
    for x in range(8):
        for y in range(8):
            board[x][y] = pos['b'][x][7-y]
            color = board[x][y].upper() == board[x][y]
            board[x][y] = board[x][y].lower() if color else board[x][y].upper()
    return {'b': board, 'c': [pos['c'][2], pos['c'][3], pos['c'][0], pos['c'][1]], 'e': [pos['e'][0], 7-pos['e'][1]] if pos['e'] is not None else None, 'w': not pos['w'], 'm': [pos['m'][0], pos['m'][1]]}

def sum(pos, func, param = None):
    sum = 0
    for x in range(8):
        for y in range(8):
            if param:
                sum += func(pos, {'x': x, 'y': y}, param)
            else:
                sum += func(pos, {'x': x, 'y': y})
    return sum

pos = {
    'b': [['r','p','-','-','-','-','P','R'],
          ['n','p','-','-','-','-','P','N'],
          ['b','p','-','-','-','-','P','B'],
          ['q','p','-','-','-','-','P','Q'],
          ['k','p','-','-','-','-','P','K'],
          ['b','p','-','-','-','-','P','B'],
          ['n','p','-','-','-','-','P','N'],
          ['r','p','-','-','-','-','P','R']],
    'c': [True, True, True, True], # castling rights 
    'e': None, #enpassant
    'w': True, # white to move?
    'm': [0, 1] # move counts
}

def scale_factor(pos, eg=None):
    if eg is None:
        eg = end_game_evaluation(pos)
    pos2 = colorflip(pos)
    pos_w = pos if eg > 0 else pos2
    pos_b = pos2 if eg > 0 else pos
    sf = 64
    pc_w = pawn_count(pos_w)
    pc_b = pawn_count(pos_b)
    qc_w = queen_count(pos_w)
    qc_b = queen_count(pos_b)
    bc_w = bishop_count(pos_w)
    bc_b = bishop_count(pos_b)
    nc_w = knight_count(pos_w)
    nc_b = knight_count(pos_b)
    npm_w = non_pawn_material(pos_w)
    npm_b = non_pawn_material(pos_b)
    bishopValueMg = 825
    bishopValueEg = 915
    rookValueMg = 1276
    if pc_w == 0 and npm_w - npm_b <= bishopValueMg:
        sf = 0 if npm_w < rookValueMg else 4 if npm_b <= bishopValueMg else 14
    if sf == 64:
        ob = opposite_bishops(pos)
        if ob and npm_w == bishopValueMg and npm_b == bishopValueMg:
            sf = 22 + 4 * candidate_passed(pos_w)
        elif ob:
            sf = 22 + 3 * piece_count(pos_w)
        else:
            if npm_w == rookValueMg and npm_b == rookValueMg and pc_w - pc_b <= 1:
                pawnking_b = 0
                pcw_flank = [0, 0]
                for x in range(8):
                    for y in range(8):
                        if board(pos_w, x, y) == "P":
                            pcw_flank[1 if x < 4 else 0] = 1
                        if board(pos_b, x, y) == "K":
                            for ix in range(-1, 2):
                                for iy in range(-1, 2):
                                    if board(pos_b, x + ix, y + iy) == "P":
                                        pawnking_b = 1
                if pcw_flank[0] != pcw_flank[1] and pawnking_b:
                    return 36
            if qc_w + qc_b == 1:
                sf = 37 + 3 * (bc_b + nc_b) if qc_w == 1 else 37 + 3 * (bc_w + nc_w)
            else:
                sf = min(sf, 36 + 7 * pc_w)
    return sf

def phase(pos):
    midgameLimit = 15258
    endgameLimit = 3915
    npm = non_pawn_material(pos) + non_pawn_material(colorflip(pos))
    npm = max(endgameLimit, min(npm, midgameLimit))
    return int(((npm - endgameLimit) * 128) / (midgameLimit - endgameLimit))

def tempo(pos, square):
    if square is not None:
        return 0
    return 28 * (1 if pos['w'] else -1)

def rule50(pos, square):
    if square is not None:
        return 0
    return pos['m'][0]


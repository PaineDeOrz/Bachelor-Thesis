#!/usr/bin/env python3
# test_my_maia_simple.py - MINIMAL test to verify your .pb.gz works

import chess.engine
import chess
import asyncio
import os

# UPDATE THESE PATHS ONLY
LC0_PATH = "../Leela_linux/build/release/lc0"  # LC0 binary
MY_MAIA_NET = "../Antrenat/maia/models/move_prediction/maia_config/ckpt-1-10000.pb.gz"  # YOUR net

async def test_single_position():
    """Test 1 position. If this works ? your net is good."""
    if not os.path.exists(LC0_PATH):
        print("? LC0 missing:", LC0_PATH)
        return
    if not os.path.exists(MY_MAIA_NET):
        print("? Your net missing:", MY_MAIA_NET)
        return
    
    print("?? Testing your Maia net...")
    transport, engine = await chess.engine.popen_uci(LC0_PATH)
    
    try:
        # Load YOUR net
        await engine.configure({"WeightsFile": MY_MAIA_NET})
        print("? Net loaded!")
        
        # Simple test position (simple pawn push)
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        print("?? Test position:", board.fen().split()[0])
        print("  Human move: e7e5")
        
        # 1 node = pure NN prediction
        result = await engine.play(board, chess.engine.Limit(nodes=1))
        engine_move = result.move.uci() if result.move else "None"
        
        print(f"? SUCCESS! Your Maia predicts: {engine_move}")
        print("?? Your .pb.gz file WORKS!")
        
    except Exception as e:
        print(f"? ERROR: {e}")
        print("?? Check net path / LC0 binary / .pb.gz integrity")
    finally:
        await engine.quit()

if __name__ == "__main__":
    asyncio.run(test_single_position())

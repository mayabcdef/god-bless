from ply2png import ply_bytes_to_grid_png_bytes
from duel_manager import duel_manager


async def compare(prompt_bytes:bytes, ply1: bytes, ply2: bytes, device: str = "cuda") -> bytes:
    img1 = ply_bytes_to_grid_png_bytes(ply1, device)
    img2 = ply_bytes_to_grid_png_bytes(ply2, device)

    duel = await duel_manager.run_duel(prompt_bytes, img1, img2)

    print("duel", duel.issues)

    score1 = duel.score1[0] + duel.score1[1]
    score2 = duel.score2[0] + duel.score2[1]

    return 1 if score1 <= score2 else 2
    
if __name__ == '__main__':
    p = '../data/ref/1.png'
    p1 = '../data/model_1/1.ply'
    p2 = '../data/model_2/1.ply'
    
    with open(p, 'rb') as f:
        prompt = f.read()
    
    with open(p1, 'rb') as f:
        ply1 = f.read()
        
    with open(p2, 'rb') as f:
        ply2 = f.read()
        
    import asyncio
    res = asyncio.run(compare(prompt, ply1, ply2))
    print(res)
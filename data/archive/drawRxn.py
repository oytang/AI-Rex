from rdkit.Chem import Draw, AllChem

def drawRxn(rxn, fileName='reaction.png'):
    rxn = AllChem.ReactionFromSmarts(rxn)
    d2d = Draw.MolDraw2DCairo(800, 300)
    d2d.DrawReaction(rxn)
    d2d.GetDrawingText()
    png = d2d.GetDrawingText()
    open(fileName, 'wb+').write(png)
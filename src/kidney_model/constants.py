"""Indices and labels used by the kidney model."""

K0 = 0  # interstitium / vessels
KD = 1  # descending tubule
KA = 2  # ascending tubule
KC = 3  # collecting tubule

COMPARTMENTS = (K0, KD, KA, KC)
COMP_NAMES = {
    K0: "0_interstitium",
    KD: "D_descending",
    KA: "A_ascending",
    KC: "C_collecting",
}

SALT = 0
UREA = 1
SOLUTES = (SALT, UREA)
SOL_NAMES = {SALT: "NaCl_salt_lumped", UREA: "urea"}

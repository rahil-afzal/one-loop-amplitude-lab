from amplitude_lab.verification import extract_final_block, verify_final_answer

LOOP = "c4**2/(32*pi**2) * (6 - log(-s/mu2) - log(-t/mu2) - log(-u/mu2))"
BETA = "-3*c4**2/(16*pi**2)"


def answer(tree="c4", loop=LOOP, beta=BETA):
    return f"Derivation...\n```final\nA_tree = {tree}\nA_loop = {loop}\nbeta_c4 = {beta}\n```\n"


def test_correct_answer_passes():
    report = verify_final_answer(answer())
    assert report.passed, report.summary()


def test_wrong_loop_constant_fails():
    report = verify_final_answer(answer(loop=LOOP.replace("6 -", "4 -")))
    assert not report.passed
    assert "loop_physical_region" in report.failed_names


def test_wrong_branch_sign_fails():
    # s-channel logarithm written as log(s/mu2) misses the +i*pi on the physical sheet
    loop = "c4**2/(32*pi**2) * (6 - log(s/mu2) - log(-t/mu2) - log(-u/mu2))"
    report = verify_final_answer(answer(loop=loop))
    assert "loop_physical_region" in report.failed_names
    assert "crossing_symmetry" in report.failed_names


def test_wrong_normalization_fails():
    report = verify_final_answer(answer(loop=LOOP.replace("32", "16")))
    assert "loop_physical_region" in report.failed_names


def test_beta_sign_error_is_reported():
    report = verify_final_answer(answer(beta="3*c4**2/(16*pi**2)"))
    assert report.failed_names == ["beta_c4"]
    assert "opposite" in next(c.message for c in report.checks if c.name == "beta_c4")


def test_missing_block_fails():
    report = verify_final_answer("no final block here")
    assert not report.passed
    assert extract_final_block("no final block here") == {}


def test_unsafe_expression_is_rejected_not_executed():
    report = verify_final_answer(answer(tree="__import__('os').getcwd()"))
    assert "parse_A_tree" in report.failed_names

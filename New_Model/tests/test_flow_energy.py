from __future__ import annotations

import torch

from fedxmag_next.flow_energy import FlowConfig, InvertibleEnergy


def test_invertibility_and_exact_jacobian() -> None:
    torch.manual_seed(12)
    model = InvertibleEnergy(FlowConfig(4, hidden=8, layers=4)).double()
    # Test nonidentity weights so a zero-initialized coupling cannot conceal a
    # sign or masking bug in its Jacobian or inverse.
    with torch.no_grad():
        for layer in model.couplings:
            layer.network[-1].weight.normal_(0, 0.05)
            layer.network[-1].bias.normal_(0, 0.02)
    values = torch.randn(5, 4, dtype=torch.float64)
    latent, logdet = model(values)
    torch.testing.assert_close(model.inverse(latent), values, atol=1e-9, rtol=1e-9)
    for row, expected in zip(values, logdet, strict=True):
        jacobian = torch.autograd.functional.jacobian(lambda x: model(x[None])[0][0], row)
        torch.testing.assert_close(torch.linalg.slogdet(jacobian)[1], expected)

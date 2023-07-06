import torch
import copy


def rep2rotmat(rep):
    # rep : bs, -1, 3, 2
    bs = rep.shape[0]
    num_jts = rep.shape[1]

    rep = rep.reshape(bs * num_jts, 3, 2)
    first_col = torch.div(rep[:, :, 0], torch.norm(rep[:, :, 0], p=2, dim=-1).reshape(-1, 1))
    second_col = torch.subtract(
        rep[:, :, 1],
        torch.sum(first_col * rep[:, :, 1], dim=-1).reshape(-1, 1) * first_col,
    )
    last_col = torch.cross(first_col, second_col)

    rotmat = torch.cat([first_col[:, :, None], second_col[:, :, None], last_col[:, :, None]], dim=-1)
    rotmat = rotmat.reshape(bs, num_jts, 3, 3)

    return rotmat


# def rep2rotmat(rep):
#     # rep : bs, -1, 3, 2
#     bs = rep.shape[0]
#     num_jts = rep.shape[1]
#     rotmat = torch.zeros(bs*num_jts, 3, 3).to(rep.device)

#     tmp_rep = rep.reshape(bs*num_jts, 3, 2)
#     rotmat[:, :, 0] = tmp_rep[:, :, 0] / torch.norm(tmp_rep[:, :, 0], p=2, dim=-1).reshape(-1, 1)
#     tmp_gs = torch.sum(rotmat[:, :, 0] * tmp_rep[:, :, 1], dim=-1)
#     rotmat[:, :, 1] = tmp_rep[:, :, 1] - tmp_gs.reshape(-1, 1)*rotmat[:, :, 0]
#     rotmat[:, :, 2] = torch.cross(rotmat[:, :, 0], rotmat[:, :, 1])

#     rotmat = rotmat.reshape(bs, num_jts, 3, 3)

#     return rotmat


def get_geodesic_loss(pred, gt, eps=1e-7):
    m = torch.bmm(pred.reshape(-1, 3, 3), gt.reshape(-1, 3, 3).transpose(1, 2))
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    theta = torch.acos(torch.clamp(cos, -1 + eps, 1 - eps))

    loss = torch.mean(theta)

    return loss

function [avPre, avRec] = evaluate_macro(Rel, Ret)

test_num   = size(Rel,2);
pre = zeros(1,test_num);
rec = zeros(1,test_num);

RetRel = (Rel & Ret);

for j = 1:test_num
    RetRel_num    = nnz(RetRel(:,j));
    Ret_num = nnz(Ret(:,j));
    Rel_num = nnz(Rel(:,j));
    if Ret_num
        pre(j) = RetRel_num / Ret_num;
    else
        pre(j) = 0;
    end
    if Rel_num
        rec(j) = RetRel_num / Rel_num;
    else
        rec(j) = 0;
    end
end

avPre = mean(pre);
avRec = mean(rec);

end

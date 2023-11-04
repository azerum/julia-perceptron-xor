function _modify!(op, a, bs...)
    ex = quote end
    bs_with_index = enumerate(bs)

    for name in fieldnames(a)
        args = map(bs_with_index) do (i, b)
            if b <: Number
                :(bs[$i])
            else
                :(bs[$i].$name)
            end
        end

        set_ex = quote a.$name = op(a.$name, $(args...)) end

        type = fieldtype(a, name)

        if !(type <: Number)
            set_ex = :(@. $set_ex)
        end

        ex = :($ex; $set_ex)
    end

    :($ex; a)
end

# `modify!(op, a, bs...)` is like `@. a = op(a, bs...)`, but for structs
@generated function modify!(op, a, bs...)
    _modify!(op, a, bs...)
end

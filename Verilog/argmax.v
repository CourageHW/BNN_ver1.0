module argmax (
    input wire clk,
    input wire rst_n,
    input wire i_valid,
    input wire [99:0] i_data,  // 10 x 10bit
    output reg [3:0] o_idx,
    output reg o_valid
);
    integer i;
    reg [9:0] cur_val;
    reg [9:0] cur_max;
    reg [3:0] class_idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_idx <= 0;
            o_valid <= 0;
        end else begin
            if (i_valid) begin
                cur_max = i_data[9:0];
                class_idx = 0;
                for (i = 1; i < 10; i = i + 1) begin
                    cur_val = i_data[i*10 +: 10];
                    if (cur_val > cur_max) begin
                        cur_max = cur_val;
                        class_idx = i[3:0];
                    end
                end
                o_idx <= class_idx;
                o_valid <= 1;
            end else begin
                o_valid <= 0;
            end
        end
    end
endmodule
